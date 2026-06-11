# models/condition_tokens.py
"""
조건 토큰 모듈.

왜곡 토큰 (DistortionToken): FOV, 렌즈 왜곡 계수 → d_model 벡터.
면적 조건은 VGGT 측에서 처리하므로 이 모델에서는 사용하지 않음.
"""

import torch
import torch.nn as nn


class DistortionToken(nn.Module):
    """
    광각 왜곡 파라미터 → d_model 차원 토큰.

    입력: [fov/180, k1, k2, cx, cy]  shape: (B, 5)
    출력: (B, 1, d_model)
    """
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.type_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.type_embed, std=0.02)

    def forward(self, distortion_params: torch.Tensor) -> torch.Tensor:
        """distortion_params: (B, 5) → (B, 1, d_model)"""
        tok = self.mlp(distortion_params).unsqueeze(1)
        return tok + self.type_embed


class ConditionTokenizer(nn.Module):
    """
    왜곡 토큰 1개를 반환.
    출력: (B, 1, d_model)
    """
    def __init__(self, cfg):
        super().__init__()
        tcfg = cfg["transformer"]
        d = tcfg["d_model"]
        self.distortion_tok = DistortionToken(tcfg["distortion_dim"], d)

    def forward(self, cond: dict) -> torch.Tensor:
        """
        cond: {"distortion": (B, 5)}
        returns: (B, 1, d_model)
        """
        return self.distortion_tok(cond["distortion"])
