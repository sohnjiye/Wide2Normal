# extract_perspectives.py
"""
파노라마 이미지에서 표준(perspective) 이미지를 추출하는 스크립트.

사용 예시:
  # 전체 데이터셋에서 추출
  python extract_perspectives.py

  # 특정 씬만
  python extract_perspectives.py --scenes scene_00001 scene_00002

  # 방향 수, 해상도 조정
  python extract_perspectives.py --n_angles 8 --size 512
"""

import os
import math
import argparse
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.utils as vutils

from data.pano_utils import equirect_to_perspective


def extract_scene(scene_path: str, output_dir: str, n_angles: int, size: int, fov: float):
    """씬 하나에서 모든 뷰포인트의 perspective 이미지를 추출."""
    scene_id = os.path.basename(scene_path)
    vp_root   = os.path.join(scene_path, "viewpoints")

    if not os.path.isdir(vp_root):
        return 0

    saved = 0
    for vp_id in sorted(os.listdir(vp_root)):
        pano_path = os.path.join(vp_root, vp_id, "panoImage_1600.jpg")
        if not os.path.exists(pano_path):
            continue

        pano   = Image.open(pano_path).convert("RGB")
        pano_t = T.ToTensor()(pano) * 2 - 1  # [-1, 1]

        out_vp_dir = os.path.join(output_dir, scene_id, vp_id)
        os.makedirs(out_vp_dir, exist_ok=True)

        for i in range(n_angles):
            theta = math.radians(360 / n_angles * i)  # 균등 간격
            crop  = equirect_to_perspective(pano_t, fov, theta, 0.0, size, size)

            angle_deg = int(360 / n_angles * i)
            out_path  = os.path.join(out_vp_dir, f"perspective_{angle_deg:03d}deg.jpg")
            vutils.save_image(crop, out_path, normalize=True, value_range=(-1, 1))
            saved += 1

    return saved


def main(args):
    data_root  = args.data_root
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 씬 목록 결정
    if args.scenes:
        scene_ids = args.scenes
    else:
        scene_ids = sorted(
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d)) and d.startswith("scene_")
        )

    print(f"추출 설정: FOV={args.fov}°, {args.n_angles}방향, {args.size}×{args.size}px")
    print(f"씬 수: {len(scene_ids)}개 → 저장 위치: {output_dir}\n")

    total = 0
    for scene_id in scene_ids:
        scene_path = os.path.join(data_root, scene_id)
        n = extract_scene(scene_path, output_dir, args.n_angles, args.size, args.fov)
        print(f"  {scene_id}: {n}장 저장")
        total += n

    print(f"\n완료: 총 {total:,}장 저장 → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  default="/home/awear/Downloads/real_world_data_wideangle")
    parser.add_argument("--output_dir", default="/home/awear/wide2normal/output/perspectives")
    parser.add_argument("--scenes",     nargs="+", default=None,
                        help="특정 씬만 추출 (미입력 시 전체)")
    parser.add_argument("--n_angles",   type=int,   default=4,
                        help="추출 방향 수 (기본 4: 0°/90°/180°/270°)")
    parser.add_argument("--size",       type=int,   default=512,
                        help="출력 이미지 해상도 (기본 512×512)")
    parser.add_argument("--fov",        type=float, default=90.0,
                        help="수평 FOV (기본 90°)")
    args = parser.parse_args()
    main(args)
