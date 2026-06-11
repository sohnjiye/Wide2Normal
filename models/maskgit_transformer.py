# models/maskgit_transformer.py
"""
Stage 2: MaskGIT Transformer

광각 이미지 feature + 조건 토큰을 바탕으로
마스킹된 이미지 토큰을 복원하는 메인 모델.

시퀀스 구조:
  [DIST_TOK] [AREA_TOK] [img_tok_0] [img_tok_1] ... [img_tok_L-1]
  ←── 2개 조건 토큰 ──→ ←─────────── L개 이미지 토큰 ────────────→

학습: 랜덤 마스킹 → CE loss (마스크 위치만)
추론: 전부 마스킹 → T 스텝에 걸쳐 confidence 순서로 채워 나감
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from .condition_tokens import ConditionTokenizer


# ── Wide-angle Image Encoder ─────────────────────────────────────────────────

class WideAngleEncoder(nn.Module):
    """
    광각 이미지(equirectangular)를 feature sequence로 인코딩.

    ViT-B/16 backbone (224×224 입력, 14×14=196 spatial tokens) 사용.
    CLS 토큰 제거 후 VQGAN 토큰 그리드(16×16=256)에 맞춰 bilinear upsample.
    출력 shape: (B, L_vq=256, d_model)
    """
    def __init__(self, d_model: int, image_size: int = 256, vq_stride: int = 16):
        super().__init__()
        vit = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.DEFAULT)
        # 분류 헤드 제거, feature extractor로만 사용
        self.conv_proj    = vit.conv_proj     # 3→768, stride 16
        self.class_token  = vit.class_token   # (1, 1, 768)
        self.vit_encoder  = vit.encoder       # positional embed 포함
        vit_dim = 768

        self.proj = nn.Linear(vit_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

        self.H_vit = 14              # ViT-B/16: 224//16 = 14 → 14×14 = 196 tokens
        self.H_vq  = image_size // vq_stride   # 256//16 = 16
        self.L     = self.H_vq ** 2            # 256

    def forward(self, wide_img: torch.Tensor) -> torch.Tensor:
        """
        wide_img: (B, 3, H, W)
        returns:  (B, L_vq, d_model)
        """
        B = wide_img.shape[0]

        # ViT는 224×224 고정 — 입력 리사이즈
        x = F.interpolate(wide_img, size=(224, 224), mode="bilinear", align_corners=False)

        # Patch embedding → (B, 768, 14, 14)
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)           # (B, 196, 768)

        # CLS 토큰 추가 (ViT encoder가 요구)
        cls = self.class_token.expand(B, -1, -1)   # (B, 1, 768)
        x   = torch.cat([cls, x], dim=1)           # (B, 197, 768)

        # Transformer encoder (positional embedding 내장)
        x = self.vit_encoder(x)                    # (B, 197, 768)

        # CLS 토큰 제거, 공간 feature만 사용
        x = x[:, 1:, :]                            # (B, 196, 768) = 14×14

        x = self.norm(self.proj(x))                # (B, 196, d_model)

        # 14×14 → 64×64 spatial upsample (VQGAN 토큰 해상도에 맞춤)
        d = x.shape[-1]
        x = x.transpose(1, 2).reshape(B, d, self.H_vit, self.H_vit)
        x = F.interpolate(x, size=(self.H_vq, self.H_vq),
                          mode="bilinear", align_corners=False)
        x = x.reshape(B, d, -1).transpose(1, 2)   # (B, L_vq, d_model)
        return x


# ── MaskGIT Transformer ───────────────────────────────────────────────────────

class MaskGITTransformer(nn.Module):
    """
    핵심 모델. 학습과 추론 모두 담당.

    Parameters
    ----------
    cfg : dict  — config.yaml 전체
    vqgan : VQGAN  — Stage 1에서 학습한 frozen VQGAN (encode/decode용)
    """
    def __init__(self, cfg, vqgan):
        super().__init__()
        tcfg = cfg["transformer"]
        self.d_model = tcfg["d_model"]
        self.n_heads = tcfg["n_heads"]
        self.n_layers = tcfg["n_layers"]
        self.vocab_size = cfg["vqgan"]["codebook_size"]
        self.mask_token_id = tcfg["mask_token_id"]   # = codebook_size (별도 토큰)
        self.num_cond = tcfg["num_cond_tokens"]       # 1 (왜곡 토큰만)

        # ── 서브모듈 ──
        img_size   = cfg["data"]["image_size"]
        vq_stride  = cfg["data"]["vq_stride"]    # VQGAN 다운샘플 배율 (= 16)
        self.wide_encoder    = WideAngleEncoder(self.d_model, img_size, vq_stride)
        self.cond_tokenizer  = ConditionTokenizer(cfg)

        # 이미지 토큰 임베딩: vocab_size+1 (MASK 토큰 포함)
        self.tok_embed = nn.Embedding(self.vocab_size + 1, self.d_model)

        # 위치 임베딩 (이미지 토큰 위치용)
        self.L = (img_size // vq_stride) ** 2    # (256//16)^2 = 256
        self.pos_embed = nn.Parameter(torch.zeros(1, self.L, self.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer (BERT-style bidirectional)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=tcfg["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN: 학습 안정성 ↑
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, self.n_layers)

        # 최종 분류 헤드: 각 위치에서 vocab_size 중 하나 예측
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.vocab_size),
        )

        # VQGAN은 frozen (codebook 변경 안 함)
        self.vqgan = vqgan
        for p in self.vqgan.parameters():
            p.requires_grad_(False)

    # ── Masking schedule ───────────────────────────────────────────────────

    @staticmethod
    def cosine_schedule(t: float) -> float:
        """γ(t) = cos(t/T × π/2)  →  [0, T] 중 랜덤 t에서 마스킹 비율"""
        return math.cos(t * math.pi / 2)

    def sample_mask(self, B: int, L: int, device) -> torch.BoolTensor:
        """
        학습 시 마스킹 비율 γ를 cosine schedule에서 랜덤 샘플링.
        반환: (B, L)  True = 마스킹된 위치
        """
        # γ를 [0, 1] 균등에서 샘플하되, cosine schedule로 변환
        r = torch.rand(B, device=device)                       # (B,)
        gamma = torch.cos(r * math.pi / 2)                    # cosine schedule
        # 각 샘플마다 gamma[b] 비율의 위치를 마스킹
        noise = torch.rand(B, L, device=device)
        mask = noise < gamma.unsqueeze(1)                      # (B, L)
        return mask

    # ── Forward (학습) ────────────────────────────────────────────────────

    def forward(self, wide_img, normal_img, cond):
        """
        학습 forward pass.

        Inputs
        ------
        wide_img   : (B, 3, H, W)  광각 이미지
        normal_img : (B, 3, H, W)  GT 표준 이미지
        cond       : {"distortion": (B, 5)}

        Returns
        -------
        loss : scalar
        """
        B = wide_img.shape[0]
        device = wide_img.device

        # 1) GT 이미지를 VQ 토큰으로 변환 (gradient 필요 없음)
        with torch.no_grad():
            _, gt_indices, _ = self.vqgan.encode(normal_img)  # (B, L)

        # 2) 랜덤 마스킹
        mask = self.sample_mask(B, self.L, device)             # (B, L) bool
        input_indices = gt_indices.clone()
        input_indices[mask] = self.mask_token_id              # [MASK] 토큰으로 교체

        # 3) 시퀀스 구성 → CE loss (마스킹된 위치만)
        logits = self._forward_tokens(wide_img, input_indices, cond)  # (B, L, V)
        loss = F.cross_entropy(
            logits[mask],        # (N_masked, V)
            gt_indices[mask],    # (N_masked,)
        )
        return loss, {"ce": loss.item()}

    def _forward_tokens(self, wide_img, input_indices, cond):
        """
        wide_img feature + 왜곡 조건 토큰 + 이미지 토큰 → logits

        시퀀스: [dist_tok(1)] + [img_tok(L)]
        """
        B = wide_img.shape[0]

        # 광각 이미지 feature (B, L, d)
        wide_feat = self.wide_encoder(wide_img)       # (B, L, d_model)

        # 왜곡 조건 토큰 (B, 1, d)
        cond_toks = self.cond_tokenizer(cond)

        # 이미지 토큰 임베딩 + 위치 임베딩 + 광각 feature 가산
        img_toks = self.tok_embed(input_indices)      # (B, L, d_model)
        img_toks = img_toks + self.pos_embed          # 위치 인코딩
        img_toks = img_toks + wide_feat               # 광각 조건 융합 ★

        # 전체 시퀀스: [cond(1) | img(L)]
        seq = torch.cat([cond_toks, img_toks], dim=1)  # (B, 1+L, d_model)

        out = self.transformer(seq)                    # (B, 1+L, d_model)

        # 이미지 토큰 위치 logits만 반환
        img_out = out[:, self.num_cond:, :]            # (B, L, d_model)
        return self.head(img_out)                      # (B, L, V)

    # ── Inference: iterative decoding ────────────────────────────────────

    @torch.no_grad()
    def generate(self, wide_img, cond, num_steps: int = 12, temperature: float = 1.0):
        """
        MaskGIT iterative decoding.

        1. 전부 [MASK]로 시작
        2. T 스텝에 걸쳐 confidence 높은 위치부터 순서대로 채움
        3. 완성된 토큰 인덱스를 VQGAN 디코더로 이미지로 복원

        Returns
        -------
        recon_img : (B, 3, H, W)  복원된 표준 이미지
        """
        B = wide_img.shape[0]
        device = wide_img.device
        H_tok = W_tok = int(self.L ** 0.5)

        # 전부 MASK로 초기화
        tokens = torch.full((B, self.L), self.mask_token_id,
                            dtype=torch.long, device=device)
        is_masked = torch.ones(B, self.L, dtype=torch.bool, device=device)

        for step in range(num_steps):
            # 1) 현재 tokens로 logits 계산
            logits = self._forward_tokens(wide_img, tokens, cond)  # (B, L, V)

            # 2) temperature scaling → 분포에서 샘플링 (argmax ❌, multinomial ✅)
            logits_scaled = logits / temperature
            probs = F.softmax(logits_scaled, dim=-1)                # (B, L, V)

            # 분포에서 토큰 샘플링
            sampled = torch.multinomial(
                probs.reshape(B * self.L, self.vocab_size), 1
            ).reshape(B, self.L)                                    # (B, L)

            # 3) 샘플링된 토큰의 확률 = confidence
            confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)  # (B, L)

            # 마스킹 안 된 위치의 confidence는 무한대 (이미 확정됨)
            confidence[~is_masked] = float("inf")

            # 4) 이번 스텝에서 몇 개 채울지: cosine schedule
            ratio = self.cosine_schedule((step + 1) / num_steps)
            n_masked = is_masked.sum(1)                             # (B,)
            n_to_keep_masked = (ratio * n_masked.float()).long()    # (B,)

            # 5) confidence 낮은 순(= 아직 불확실한 순)으로 mask 유지
            #    → 높은 confidence 위치부터 채워 나감
            sorted_conf = confidence.argsort(dim=1)                 # (B, L) 오름차순
            new_mask = torch.zeros_like(is_masked)
            for b in range(B):
                keep = n_to_keep_masked[b].item()
                if keep > 0:
                    new_mask[b, sorted_conf[b, :keep]] = True

            # 6) 확정 위치 업데이트
            tokens = torch.where(is_masked & ~new_mask, sampled, tokens)
            is_masked = new_mask

        # VQGAN 디코더로 이미지 복원
        recon_img = self.vqgan.decode_from_indices(tokens, H_tok, W_tok)
        return recon_img
