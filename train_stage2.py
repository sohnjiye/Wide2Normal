# train_stage2.py
"""
Stage 2 실행: MaskGIT Transformer 학습

python train_stage2.py --config configs/config.yaml \
                       --vqgan_ckpt checkpoints/stage1/vqgan-best.ckpt
"""

import argparse
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataset import build_dataloader
from training.lightning_module import MaskGITModule


def main(cfg, vqgan_ckpt):
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("high")

    # 데이터 (Stage 2: wide + normal + cond 전부)
    train_dl = build_dataloader(cfg, split="train", stage=2)
    val_dl   = build_dataloader(cfg, split="val",   stage=2)

    # 모델 (VQGAN frozen 자동으로 처리됨)
    model = MaskGITModule(cfg, vqgan_ckpt_path=vqgan_ckpt)

    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/stage2",
            filename="maskgit-epoch={epoch:03d}-valloss={val/loss:.4f}",
            monitor="val/loss",
            save_top_k=3,
            mode="min",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg["training"]["stage2_epochs"],
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        gradient_clip_val=cfg["training"]["grad_clip"],
        log_every_n_steps=cfg["training"]["log_every_n_steps"],
        val_check_interval=cfg["training"]["val_check_interval"],
        callbacks=callbacks,
        logger=TensorBoardLogger("logs", name="stage2_maskgit"),
    )

    trainer.fit(model, train_dl, val_dl, ckpt_path=args.resume_ckpt)
    print(f"\nStage 2 완료. 체크포인트: checkpoints/stage2/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--vqgan_ckpt",  required=True,
                        help="Stage 1 VQGAN 체크포인트 경로")
    parser.add_argument("--resume_ckpt", default=None,
                        help="이어서 학습할 Stage 2 체크포인트 경로 (선택)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.vqgan_ckpt)
