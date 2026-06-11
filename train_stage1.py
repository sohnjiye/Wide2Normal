# train_stage1.py
"""
Stage 1 실행: VQGAN codebook 학습

python train_stage1.py --config configs/config.yaml
"""

import argparse
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataset import build_dataloader
from training.lightning_module import VQGANModule


def main(cfg):
    pl.seed_everything(42)

    # Tensor Core 활성화 (RTX 시리즈 속도 2~3배 향상)
    torch.set_float32_matmul_precision("high")

    # 데이터 (Stage 1: normal 이미지만)
    train_dl = build_dataloader(cfg, split="train", stage=1)
    val_dl   = build_dataloader(cfg, split="val",   stage=1)

    # 모델
    model = VQGANModule(cfg)

    # 콜백
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/stage1",
            filename="vqgan-epoch={epoch:03d}-valloss={val/loss:.4f}",
            monitor="val/loss",
            save_top_k=3,
            mode="min",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg["training"]["stage1_epochs"],
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        gradient_clip_val=cfg["training"]["grad_clip"],
        log_every_n_steps=cfg["training"]["log_every_n_steps"],
        val_check_interval=cfg["training"]["val_check_interval"],
        callbacks=callbacks,
        logger=TensorBoardLogger("logs", name="stage1_vqgan"),
    )

    trainer.fit(model, train_dl, val_dl)
    print(f"\nStage 1 완료. 체크포인트: checkpoints/stage1/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
