#!/bin/bash


# 환경변수 설정
export WANDB_ENTITY="comflife" 

# 학습 명령어 - yolov5s
echo "=== YOLOv5x RGBT 모델 학습 시작 ==="
python3 train_simple.py \
    --img 640 \
    --batch-size 12 \
    --epochs 30 \
    --data data/kaist-rgbt.yaml \
    --cfg models/yolov5x_kaist-rgbt.yaml \
    --weights yolov5x.pt \
    --workers 8 \
    --name yolov5x-rgbt-new191 \
    --rgbt \
    --hyp data/hyps/hyp.scratch-rgbt.yaml \
    --entity $WANDB_ENTITY \
    --project pedestrian \
    --quad \
    --evolve \
    --device 0 \
    --noval \
    --image-weights \
    --multi-scale

# 검증 명령어 - yolov5s
echo "=== YOLOv5x RGBT 모델 검증 시작 ==="
python3 val.py \
    --weights /home/byounggun/AUE8088/pedestrian/yolov5x-rgbt-new191/weights/best.pt \
    --data data/kaist-rgbt.yaml \
    --task test \
    --save-json \
    --img 640 \
    --rgbt_input



echo "=== 학습 및 검증 완료 ==="
