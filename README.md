# Wide2Normal — 광각→표준 이미지 변환 모델

광각(wide-angle) 실내 사진을 표준 화각(perspective, FOV 90°) 이미지로 변환하는 VQGAN + MaskGIT Transformer 기반 모델입니다. RealSee3D 등의 파노라마 데이터셋에서 광각-표준 이미지 쌍을 생성해 학습합니다.

## 환경 설정

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision pytorch-lightning pandas pillow pyyaml
```

> Real-ESRGAN 사전학습 가중치(`models/weights/realesr-general-x4v3.pth`)는 [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases)에서 받을 수 있습니다 (저장소에 포함되어 있음).

## 프로젝트 구조

```
wide2normal/
├── configs/
│   └── config.yaml          # 모든 하이퍼파라미터
├── data/
│   └── dataset.py           # RealSee3D 데이터셋 + 전처리
├── models/
│   ├── vqgan.py             # Stage 1: VQGAN (codebook 학습)
│   ├── condition_tokens.py  # 왜곡 토큰 + 면적 조건 토큰
│   └── maskgit_transformer.py  # Stage 2: MaskGIT Transformer
├── training/
│   └── lightning_module.py  # PyTorch Lightning 학습 모듈
├── train_stage1.py          # VQGAN 학습 실행
├── train_stage2.py          # Transformer 학습 실행
└── inference.py             # 추론 스크립트
```

## 데이터 준비

```bash
# 파노라마 데이터에서 광각-표준 이미지 쌍 메타데이터 생성
python data/prepare_metadata.py \
  --data_root /path/to/real_world_data_wideangle \
  --output    data/metadata.csv
```

`configs/config.yaml`의 `data.data_root` / `data.metadata_path`를 환경에 맞게 수정하세요.

## 학습 순서

```bash
# Stage 1: VQGAN codebook 학습 (표준 이미지만 사용)
python train_stage1.py --config configs/config.yaml

# Stage 2: MaskGIT Transformer 학습 (광각→표준, codebook frozen)
python train_stage2.py --config configs/config.yaml
```

체크포인트는 `checkpoints/stage1/`, `checkpoints/stage2/`에 저장되며 용량 문제로 저장소에는 포함되어 있지 않습니다.

## 추론

```bash
python inference.py \
  --config     configs/config.yaml \
  --ckpt       checkpoints/stage2/maskgit-epoch=128-valloss=1.2354.ckpt \
  --vqgan_ckpt checkpoints/stage1/vqgan-epoch=011-val/loss=0.1119.ckpt \
  --n_samples  8 \
  --output     output/inference
```

## 핵심 아이디어

1. **VQGAN**: 표준 이미지를 이산 토큰으로 압축하는 codebook 구축
2. **조건 토큰**: 광각 왜곡 파라미터(FOV, k1/k2) + 면적 정보(평수)를 벡터로 임베딩
3. **MaskGIT**: 광각 feature를 조건으로, 마스킹된 이미지 토큰을 복원하도록 학습
4. **Iterative decoding**: 추론 시 confidence 순서대로 토큰을 채워 나감
