# data/pano_utils.py
"""
Equirectangular panorama → Perspective projection.

equirect_to_perspective():
  주어진 방위각/앙각과 FOV로 equirect 파노라마에서 perspective crop을 추출.
  F.grid_sample 기반으로 미분 가능하며 배치 사용 가능.

좌표계:
  - 경도(lon) : [-π, π]   좌-우 (동-서 방향)
  - 위도(lat) : [-π/2, π/2]  상-하 (북극-남극)
  - equirect 픽셀 : 상단=lat+90°, 하단=lat-90°
"""

import math
import torch
import torch.nn.functional as F


def equirect_to_perspective(
    pano: torch.Tensor,
    fov_h_deg: float,
    theta: float,
    phi: float,
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    """
    Equirectangular → Perspective projection (CPU/GPU 호환).

    Parameters
    ----------
    pano      : (3, H, W) float tensor, [-1, 1] 정규화됨
    fov_h_deg : 수평 FOV (도). 90.0 = 표준 perspective, 120+ = 광각
    theta     : 수평 회전 (yaw, 라디안). 0=정면, +π/2=오른쪽
    phi       : 수직 기울기 (pitch, 라디안). 0=수평, +0.1=약간 위
    out_h     : 출력 높이 (픽셀)
    out_w     : 출력 너비 (픽셀)

    Returns
    -------
    (3, out_h, out_w) float tensor, [-1, 1]
    """
    device = pano.device

    fov_h = math.radians(fov_h_deg)
    fov_v = fov_h * out_h / out_w  # square pixels 가정

    tan_h = math.tan(fov_h / 2)
    tan_v = math.tan(fov_v / 2)

    # 카메라 공간 ray 방향 그리드 (Z 앞방향, X 오른쪽, Y 위)
    xs = torch.linspace(-tan_h, tan_h, out_w, device=device)
    ys = torch.linspace(tan_v, -tan_v, out_h, device=device)  # top-to-bottom
    yv, xv = torch.meshgrid(ys, xs, indexing="ij")             # (out_h, out_w)
    zv = torch.ones_like(xv)

    # 단위 벡터로 정규화
    inv_len = 1.0 / torch.sqrt(xv ** 2 + yv ** 2 + zv ** 2)
    dx, dy, dz = xv * inv_len, yv * inv_len, zv * inv_len

    # Yaw 회전 (Y축 기준, theta)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx_r =  cos_t * dx + sin_t * dz
    dy_r =  dy
    dz_r = -sin_t * dx + cos_t * dz

    # Pitch 회전 (X축 기준, phi)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    dx_f =  dx_r
    dy_f =  cos_p * dy_r - sin_p * dz_r
    dz_f =  sin_p * dy_r + cos_p * dz_r

    # 구면 좌표 변환
    lon = torch.atan2(dx_f, dz_f)                           # [-π, π]
    lat = torch.asin(dy_f.clamp(-1 + 1e-6, 1 - 1e-6))      # [-π/2, π/2]

    # F.grid_sample 좌표: [-1, 1]
    # lon: -π→-1, +π→+1 (수평 wrap-around 처리됨)
    # lat: +π/2→-1(상단픽셀), -π/2→+1(하단픽셀) — equirect 상단=북극이므로 부호 반전
    grid_x = lon / math.pi
    grid_y = -lat / (math.pi / 2)

    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    out = F.grid_sample(
        pano.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return out.squeeze(0)  # (3, out_h, out_w)
