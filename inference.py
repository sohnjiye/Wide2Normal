# inference.py
"""
학습된 MaskGIT 모델로 추론 실행.

사용:
  python inference.py \
    --config    configs/config.yaml \
    --ckpt      checkpoints/stage2/maskgit-epoch=128-valloss=1.2354.ckpt \
    --vqgan_ckpt checkpoints/stage1/vqgan-epoch=011-val/loss=0.1119.ckpt \
    --n_samples 8 \
    --output    output/inference
"""

import argparse
import os
import math
import yaml
import torch
import torchvision.utils as vutils
import pandas as pd
from PIL import Image
import torchvision.transforms as T

from models.vqgan import VQGAN
from models.maskgit_transformer import MaskGITTransformer
from data.pano_utils import equirect_to_perspective


def load_model(cfg, ckpt_path, vqgan_ckpt_path, device):
    vqgan = VQGAN(cfg)
    vqgan_ckpt = torch.load(vqgan_ckpt_path, map_location="cpu")
    state = {k.replace("vqgan.", ""): v
             for k, v in vqgan_ckpt["state_dict"].items() if k.startswith("vqgan.")}
    vqgan.load_state_dict(state)
    vqgan.eval()

    model = MaskGITTransformer(cfg, vqgan)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_state = {k.replace("model.", ""): v
                   for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(model_state)
    model.eval()

    return model.to(device)


def build_cond(batch_size, device):
    distortion = torch.tensor([
        105.0 / 180.0, -0.3, 0.1, 0.5, 0.5,
    ], dtype=torch.float32).unsqueeze(0).expand(batch_size, -1).to(device)
    return {"distortion": distortion}


@torch.no_grad()
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = load_model(cfg, args.ckpt, args.vqgan_ckpt, device)
    print(f"모델 로드 완료: {args.ckpt}")

    image_size = cfg["data"]["image_size"]
    fov = cfg["data"].get("perspective_fov", 90.0)

    wide_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    meta = pd.read_csv(cfg["data"]["metadata_path"])
    samples = meta.sample(n=args.n_samples, random_state=42)

    os.makedirs(args.output, exist_ok=True)

    all_wide, all_gt, all_recon = [], [], []

    for idx, row in samples.iterrows():
        wide_img = Image.open(str(row["wide_path"])).convert("RGB")
        wide = wide_transform(wide_img).unsqueeze(0).to(device)

        pano_img = Image.open(str(row["pano_path"])).convert("RGB")
        pano_t = T.ToTensor()(pano_img) * 2.0 - 1.0
        yaw_rad = math.radians(float(row["yaw_deg"]))
        gt = equirect_to_perspective(
            pano_t, fov_h_deg=fov, theta=yaw_rad, phi=0.0,
            out_h=image_size, out_w=image_size,
        ).unsqueeze(0).to(device)

        cond = build_cond(1, device)
        recon = model.generate(
            wide, cond,
            num_steps=cfg["inference"]["num_steps"],
            temperature=cfg["inference"]["temperature"],
        )

        all_wide.append(wide.cpu())
        all_gt.append(gt.cpu())
        all_recon.append(recon.cpu())

        print(f"  샘플 {len(all_wide)}/{args.n_samples} — scene={row['scene_id']} yaw={row['yaw_deg']}°")

    wide_all  = torch.cat(all_wide)
    gt_all    = torch.cat(all_gt)
    recon_all = torch.cat(all_recon)

    interleaved = torch.stack([wide_all, gt_all, recon_all], dim=1).reshape(-1, 3, image_size, image_size)
    grid = vutils.make_grid(interleaved, nrow=3, normalize=True, value_range=(-1, 1), padding=4)
    grid_path = os.path.join(args.output, "comparison_grid.png")
    vutils.save_image(grid, grid_path)
    print(f"\n비교 그리드 저장: {grid_path}")
    print("열 순서: [광각 입력 | GT 표준 | MaskGIT 출력]")

    for i in range(args.n_samples):
        yaw = int(samples.iloc[i]["yaw_deg"])
        scene = samples.iloc[i]["scene_id"]
        vutils.save_image(wide_all[i],  os.path.join(args.output, f"{i:02d}_{scene}_yaw{yaw}_wide.jpg"),
                          normalize=True, value_range=(-1, 1))
        vutils.save_image(gt_all[i],    os.path.join(args.output, f"{i:02d}_{scene}_yaw{yaw}_gt.jpg"),
                          normalize=True, value_range=(-1, 1))
        vutils.save_image(recon_all[i], os.path.join(args.output, f"{i:02d}_{scene}_yaw{yaw}_recon.jpg"),
                          normalize=True, value_range=(-1, 1))

    print(f"개별 이미지 저장 완료: {args.output}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--ckpt",        required=True, help="Stage 2 체크포인트")
    parser.add_argument("--vqgan_ckpt",  required=True, help="Stage 1 VQGAN 체크포인트")
    parser.add_argument("--n_samples",   type=int, default=8)
    parser.add_argument("--output",      default="output/inference")
    args = parser.parse_args()
    run_inference(args)
