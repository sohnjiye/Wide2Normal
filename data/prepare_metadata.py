# data/prepare_metadata.py
"""
RealSee3D 데이터셋 메타데이터 CSV 생성 스크립트.

학습 쌍:
  wide  (입력) : wideangle/photo_N_yawXXX_rgb.jpg  (실제 광각 이미지)
  normal (GT)  : panoImage_1600.jpg 에서 같은 yaw 방향으로 90° FOV perspective crop

사용:
  python data/prepare_metadata.py \
    --data_root /home/awear/Downloads/real_world_data_wideangle \
    --output    /home/awear/wide2normal/data/metadata.csv
"""

import os
import re
import argparse
import random
import pandas as pd


def parse_yaw(filename: str):
    """photo_N_yawXXX_rgb.jpg 에서 yaw 각도(float) 추출."""
    m = re.search(r"yaw(-?\d+)", filename)
    return float(m.group(1)) if m else None


def main(data_root: str, output_path: str) -> None:
    random.seed(42)

    scene_dirs = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d)) and d.startswith("scene_")
    )

    rows = []
    for scene_id in scene_dirs:
        vp_root = os.path.join(data_root, scene_id, "viewpoints")
        if not os.path.isdir(vp_root):
            continue

        # 씬별 합성 면적 (일관성 유지)
        exclusive = round(random.uniform(33.0, 130.0), 1)
        supply    = round(exclusive * random.uniform(1.15, 1.35), 1)

        for vp_id in sorted(os.listdir(vp_root)):
            vp_path   = os.path.join(vp_root, vp_id)
            pano_path = os.path.join(vp_path, "panoImage_1600.jpg")
            wide_dir  = os.path.join(vp_path, "wideangle")

            if not os.path.exists(pano_path) or not os.path.isdir(wide_dir):
                continue

            # 층수
            floor = 1
            floor_txt = os.path.join(vp_path, "floor.txt")
            if os.path.exists(floor_txt):
                try:
                    floor = max(1, int(float(open(floor_txt).read().strip())))
                except ValueError:
                    pass

            # 광각 이미지 목록 (rgb만)
            for fname in sorted(os.listdir(wide_dir)):
                if not fname.endswith("_rgb.jpg"):
                    continue
                yaw = parse_yaw(fname)
                if yaw is None:
                    continue

                rows.append({
                    "scene_id":          scene_id,
                    "viewpoint_id":      vp_id,
                    "yaw_deg":           yaw,
                    "wide_path":         os.path.join(wide_dir, fname),
                    "pano_path":         pano_path,
                    "exclusive_area_m2": exclusive,
                    "supply_area_m2":    supply,
                    "floor":             floor,
                })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"저장 완료: {output_path}")
    print(f"  총 샘플:  {len(df):,}개")
    print(f"  씬 수:    {df['scene_id'].nunique():,}개")
    print(f"  yaw 방향: {sorted(df['yaw_deg'].unique().tolist())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output",    required=True)
    args = parser.parse_args()
    main(args.data_root, args.output)
