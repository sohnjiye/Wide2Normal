# training/lightning_module.py
"""
PyTorch Lightning 학습 모듈.

Stage 1: VQGANModule  — VQGAN codebook 학습
Stage 2: MaskGITModule — MaskGIT Transformer 학습 (VQGAN frozen)

발표 팁: trainer.fit() 한 줄로 학습 실행됨.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision

from models.vqgan import VQGAN
from models.maskgit_transformer import MaskGITTransformer


# ── Stage 1: VQGAN ──────────────────────────────────────────────────────────

class VQGANModule(pl.LightningModule):
    """
    표준 이미지만으로 codebook 학습.
    Perceptual loss 위해 VGG feature를 사용.
    """
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.vqgan = VQGAN(cfg)

        # Perceptual loss용 VGG (frozen)
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
        self.vgg_features = torch.nn.Sequential(*list(vgg.features)[:16])
        for p in self.vgg_features.parameters():
            p.requires_grad_(False)

        self.lambda_perc = cfg["vqgan"]["lambda_perceptual"]

    def forward(self, x):
        return self.vqgan(x)

    def _is_tb_logger(self) -> bool:
        """TensorBoard logger가 연결되어 있는지 확인."""
        return (self.logger is not None
                and hasattr(self.logger, "experiment")
                and hasattr(self.logger.experiment, "add_image"))

    def _perceptual_loss(self, x, x_recon):
        feat_real  = self.vgg_features(x)
        feat_recon = self.vgg_features(x_recon)
        return F.l1_loss(feat_recon, feat_real.detach())

    def training_step(self, batch, batch_idx):
        x = batch["normal"]                                  # (B, 3, H, W)
        x_recon, _, vq_loss = self.vqgan(x)

        loss_recon = F.l1_loss(x_recon, x)
        loss_perc  = self._perceptual_loss(x, x_recon)
        loss       = loss_recon + self.lambda_perc * loss_perc + vq_loss

        self.log_dict({
            "train/recon": loss_recon,
            "train/perceptual": loss_perc,
            "train/vq": vq_loss,
            "train/total": loss,
        }, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["normal"]
        x_recon, _, vq_loss = self.vqgan(x)
        loss = F.l1_loss(x_recon, x) + vq_loss
        self.log("val/loss", loss, prog_bar=True)

        # 첫 번째 배치 이미지 로깅 (TensorBoard 연결된 경우에만)
        if batch_idx == 0 and self._is_tb_logger():
            grid = torchvision.utils.make_grid(
                torch.cat([x[:4], x_recon[:4]]),
                nrow=4, normalize=True, value_range=(-1, 1)
            )
            self.logger.experiment.add_image("val/recon", grid, self.global_step)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.vqgan.parameters(),
            lr=float(self.cfg["training"]["stage1_lr"]),
            betas=(0.9, 0.95),
            weight_decay=1e-4,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg["training"]["stage1_epochs"]
        )
        return [opt], [sched]


# ── Stage 2: MaskGIT Transformer ────────────────────────────────────────────

class MaskGITModule(pl.LightningModule):
    """
    Stage 1 VQGAN을 frozen으로 불러와서
    MaskGIT Transformer만 학습.
    """
    def __init__(self, cfg, vqgan_ckpt_path: str):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # Stage 1 VQGAN 로드 & freeze
        vqgan = VQGAN(cfg)
        ckpt = torch.load(vqgan_ckpt_path, map_location="cpu")
        # Lightning checkpoint 구조 처리
        state = {k.replace("vqgan.", ""): v
                 for k, v in ckpt["state_dict"].items() if k.startswith("vqgan.")}
        vqgan.load_state_dict(state)
        vqgan.eval()

        # MaskGIT Transformer (내부에서 vqgan을 frozen으로 보유)
        self.model = MaskGITTransformer(cfg, vqgan)

    def _is_tb_logger(self) -> bool:
        """TensorBoard logger가 연결되어 있는지 확인."""
        return (self.logger is not None
                and hasattr(self.logger, "experiment")
                and hasattr(self.logger.experiment, "add_image"))

    def training_step(self, batch, batch_idx):
        wide   = batch["wide"]
        normal = batch["normal"]
        cond   = batch["cond"]

        loss, log_dict = self.model(wide, normal, cond)

        self.log_dict({
            "train/loss":    loss,
            "train/loss_ce": log_dict["ce"],
        }, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        wide   = batch["wide"]
        normal = batch["normal"]
        cond   = batch["cond"]

        loss, _ = self.model(wide, normal, cond)
        self.log("val/loss", loss, prog_bar=True)

        # 추론 이미지 로깅 (TensorBoard 연결된 경우에만)
        if batch_idx == 0 and self._is_tb_logger():
            num_steps = self.cfg["inference"]["num_steps"]
            recon = self.model.generate(wide[:4], {k: v[:4] for k, v in cond.items()},
                                        num_steps=num_steps)
            grid = torchvision.utils.make_grid(
                torch.cat([wide[:4], recon, normal[:4]]),
                nrow=4, normalize=True, value_range=(-1, 1)
            )
            self.logger.experiment.add_image("val/wide_recon_gt", grid, self.global_step)

    def configure_optimizers(self):
        # VQGAN은 frozen이므로 Transformer + 인코더 파라미터만
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params,
            lr=float(self.cfg["training"]["stage2_lr"]),
            betas=(0.9, 0.95),
            weight_decay=1e-4,
        )
        # Warmup + cosine decay
        total_steps = self.cfg["training"]["stage2_epochs"] * 1000  # 대략적인 스텝 수
        warmup_steps = total_steps // 20

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return [opt], [{"scheduler": sched, "interval": "step"}]
