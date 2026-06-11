# models/vqgan.py
"""
Stage 1: VQGAN
표준 이미지를 이산 토큰으로 압축하는 codebook을 학습.
Transformer (Stage 2)가 이 codebook 공간에서 동작하게 됨.

구조:
  Encoder → VectorQuantizer → Decoder
                ↕
           Codebook (K × d)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Residual Block ──────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


# ── Encoder ─────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    이미지 (B, 3, H, W) → feature map (B, codebook_dim, H/16, W/16)
    stride=2 conv로 4번 다운샘플 → 16배 축소
    """
    def __init__(self, channels, codebook_dim):
        super().__init__()
        layers = [nn.Conv2d(3, channels[0], 3, padding=1)]
        for i in range(len(channels) - 1):
            layers += [
                ResBlock(channels[i]),
                nn.Conv2d(channels[i], channels[i+1], 4, stride=2, padding=1),
            ]
        layers += [ResBlock(channels[-1]), ResBlock(channels[-1])]
        layers += [nn.GroupNorm(32, channels[-1]), nn.SiLU()]
        layers += [nn.Conv2d(channels[-1], codebook_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Decoder ─────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    feature map (B, codebook_dim, H/16, W/16) → 이미지 (B, 3, H, W)
    """
    def __init__(self, channels, codebook_dim):
        super().__init__()
        ch = list(reversed(channels))
        layers = [nn.Conv2d(codebook_dim, ch[0], 3, padding=1)]
        layers += [ResBlock(ch[0]), ResBlock(ch[0])]
        for i in range(len(ch) - 1):
            layers += [
                nn.ConvTranspose2d(ch[i], ch[i+1], 4, stride=2, padding=1),
                ResBlock(ch[i+1]),
            ]
        layers += [nn.GroupNorm(32, ch[-1]), nn.SiLU()]
        layers += [nn.Conv2d(ch[-1], 3, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Vector Quantizer ────────────────────────────────────────────────────────

class VectorQuantizer(nn.Module):
    """
    EMA 업데이트 방식의 VQ.
    - encoder output을 codebook의 가장 가까운 벡터로 교체
    - straight-through estimator로 gradient 통과
    - EMA로 codebook 업데이트 (더 안정적)
    """
    def __init__(self, codebook_size, codebook_dim, commitment_cost=0.25):
        super().__init__()
        self.K = codebook_size
        self.d = codebook_dim
        self.commitment_cost = commitment_cost

        # Codebook: K개의 d차원 벡터
        self.embedding = nn.Embedding(self.K, self.d)
        nn.init.uniform_(self.embedding.weight, -1/self.K, 1/self.K)

        # EMA 통계
        self.register_buffer("ema_cluster_size", torch.zeros(self.K))
        self.register_buffer("ema_embed_avg", self.embedding.weight.data.clone())
        self.ema_decay = 0.99

    def forward(self, z):
        """
        z: (B, d, H, W)
        returns:
          z_q:     quantized feature (B, d, H, W)  — decoder에 전달
          indices: token index map   (B, H*W)       — Transformer가 예측할 GT
          loss:    VQ commitment loss (scalar)
        """
        B, d, H, W = z.shape
        # (B, H, W, d) → (B*H*W, d)
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, d)

        # 거리 계산: ||z - e||² = ||z||² - 2*z·eᵀ + ||e||²
        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.T
            + self.embedding.weight.pow(2).sum(1)
        )
        indices_flat = dist.argmin(1)                  # (B*H*W,)
        z_q_flat = self.embedding(indices_flat)         # (B*H*W, d)

        # EMA codebook 업데이트 (학습 시만)
        if self.training:
            self._ema_update(z_flat, indices_flat)

        # Commitment loss: encoder output이 codebook에 가까워지도록
        loss = self.commitment_cost * F.mse_loss(z_flat, z_q_flat.detach())

        # Straight-through: gradient를 encoder로 그대로 통과
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()

        z_q = z_q_flat.reshape(B, H, W, d).permute(0, 3, 1, 2)
        indices = indices_flat.reshape(B, H * W)

        return z_q, indices, loss

    @torch.no_grad()
    def _ema_update(self, z_flat, indices_flat):
        one_hot = F.one_hot(indices_flat, self.K).float()          # (N, K)
        self.ema_cluster_size = (
            self.ema_decay * self.ema_cluster_size
            + (1 - self.ema_decay) * one_hot.sum(0)
        )
        embed_sum = one_hot.T @ z_flat                             # (K, d)
        self.ema_embed_avg = (
            self.ema_decay * self.ema_embed_avg
            + (1 - self.ema_decay) * embed_sum
        )
        # Laplace smoothing
        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + 1e-5) / (n + self.K * 1e-5) * n
        self.embedding.weight.data = self.ema_embed_avg / smoothed.unsqueeze(1)

    def indices_to_embedding(self, indices):
        """인덱스 → codebook 벡터 (추론 시 디코더로 넘길 때)"""
        return self.embedding(indices)   # (B, L, d)


# ── VQGAN (합친 모델) ────────────────────────────────────────────────────────

class VQGAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        vcfg = cfg["vqgan"]
        self.encoder  = Encoder(vcfg["encoder_channels"], vcfg["codebook_dim"])
        self.decoder  = Decoder(vcfg["decoder_channels"], vcfg["codebook_dim"])
        self.quantizer = VectorQuantizer(vcfg["codebook_size"], vcfg["codebook_dim"],
                                         vcfg["lambda_commit"])

    def encode(self, x):
        """이미지 → (quantized feature, token indices, vq_loss)"""
        z = self.encoder(x)
        return self.quantizer(z)

    def decode(self, z_q):
        """Quantized feature → 이미지"""
        return self.decoder(z_q)

    def decode_from_indices(self, indices, H_tok, W_tok):
        """
        Token 인덱스 → 이미지 (MaskGIT 추론 후 최종 복원)
        indices: (B, H_tok*W_tok)
        """
        B = indices.shape[0]
        emb = self.quantizer.indices_to_embedding(indices)    # (B, L, d)
        d = emb.shape[-1]
        z_q = emb.reshape(B, H_tok, W_tok, d).permute(0, 3, 1, 2)
        return self.decode(z_q)

    def forward(self, x):
        z_q, indices, vq_loss = self.encode(x)
        x_recon = self.decode(z_q)
        return x_recon, indices, vq_loss
