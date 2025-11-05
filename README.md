# Dual-Head Multi-Label Classification

## 🗂️ Project Architecture
```bash
project_root/
├── src/
│ ├── model.py # DualHeadClassifier
│ ├── dataset.py # MultiHeadDataset
│ └── train.py
│
├── tools/
│ └── export.py # Torch to Onnx
│
├── data/
│ ├── images/ # 입력 이미지 폴더
│ └── labels/ # 이미지와 동일 이름의 .txt 라벨 파일
│
├── checkpoints/ # 학습된 모델 및 ONNX 파일 저장 폴더
│
└── README.md
```
<br>

## 🚀 Quick Start
### 1. Docker image pull
```bash
docker pull pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel
```

### 2. Docker container run
```bash
docker run --gpus all -it --shm-size=32g --name $container_name -v $local_project_path:/workspace/$project_folder -v $local_data_path:/$project_folder/data/ docker.io/pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel
```

### 3. Data preparation
```bash
# Label format:
<HeadA Label Index>
<HeadB Label Index 3개 쉼표로 구분>

# Label example:
2
0,1,3
```

### 4. Model Training
`train.py` trains the model with the specified parameters
```bash
python3 src/train.py

python3 src/train.py \
  --data-dir data/cropped_maps \
  --label-dir data/labels \
  --ckpt-dir checkpoints \
  --epochs 40 \
  --batch-size 16 \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --val-ratio 0.2 \
  --img-size 224 \
  --num-classes 4 \
  --k-top 3 \
  --num-workers 8 \
  --device cuda \
  --seed 1337
```

### 5. Torch2ONNX
```bash
python3 tools/export.py \
  --checkpoint checkpoints/best_model_epoch28.pth \
  --onnx-output checkpoints/best_model.onnx \
  --num-classes 4 \
  --input-size 3 224 224 \
  --opset 17
```

<br>