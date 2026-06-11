# data/dataset.py
"""
RealSee3D 데이터셋 로더.

학습 쌍:
  wide   (입력) : wideangle/photo_N_yawXXX_rgb.jpg  ← 실제 광각 이미지
  normal (GT)   : panoImage_1600.jpg 에서 동일 yaw 방향으로 90° FOV perspective crop

Stage 1: normal 이미지만 (VQGAN codebook 학습)
Stage 2: wide + normal (MaskGIT Transformer 학습)
"""

import math
import random

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from data.pano_utils import equirect_to_perspective


class RealSee3DDataset(Dataset):

    def __init__(self, cfg, split: str = "train", stage: int = 2):
        self.cfg        = cfg
        self.stage      = stage
        self.split      = split
        self.image_size = cfg["data"]["image_size"]
        self.fov        = cfg["data"].get("perspective_fov", 90.0)

        self.meta = pd.read_csv(cfg["data"]["metadata_path"])

        # 80:20 train/val 분할
        idxs = list(range(len(self.meta)))
        random.seed(42)
        random.shuffle(idxs)
        cut = int(len(idxs) * 0.8)
        self.idxs = idxs[:cut] if split == "train" else idxs[cut:]

        # 광각 이미지 transform (입력)
        self.wide_transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, i: int) -> dict:
        row = self.meta.iloc[self.idxs[i]]

        # ── Normal GT: 파노라마에서 같은 yaw 방향으로 90° FOV crop ────────
        pano_img = Image.open(str(row["pano_path"])).convert("RGB")
        pano_t   = T.ToTensor()(pano_img) * 2.0 - 1.0  # [-1, 1]

        yaw_rad = math.radians(float(row["yaw_deg"]))

        # val은 고정, train은 약간의 pitch 랜덤 augmentation
        phi = random.uniform(-0.1, 0.1) if self.split == "train" else 0.0

        normal = equirect_to_perspective(
            pano_t,
            fov_h_deg=self.fov,
            theta=yaw_rad,
            phi=phi,
            out_h=self.image_size,
            out_w=self.image_size,
        )  # (3, H, W), [-1, 1]

        if self.stage == 1:
            return {"normal": normal}

        # ── Wide 입력: 실제 광각 이미지 ───────────────────────────────────
        wide_img = Image.open(str(row["wide_path"])).convert("RGB")
        wide     = self.wide_transform(wide_img)  # (3, H, W), [-1, 1]

        return {
            "wide":   wide,
            "normal": normal,
            "cond":   self._build_cond(),
        }

    def _build_cond(self) -> dict:
        # 왜곡 토큰: 광각 렌즈 특성 (고정값, 추후 실제 캘리브레이션 데이터로 교체 가능)
        distortion = torch.tensor([
            105.0 / 180.0,  # fov_norm (105° 광각)
            -0.3,           # k1 (배럴 왜곡 전형값)
            0.1,            # k2
            0.5,            # cx
            0.5,            # cy
        ], dtype=torch.float32)
        return {"distortion": distortion}


def build_dataloader(cfg, split: str = "train", stage: int = 2) -> DataLoader:
    ds = RealSee3DDataset(cfg, split=split, stage=stage)
    return DataLoader(
        ds,
        batch_size=cfg["training"][f"stage{stage}_batch_size"],
        shuffle=(split == "train"),
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=(split == "train"),
    )
