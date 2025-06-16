#!/usr/bin/env python3
import json
import os

# 파일 경로
annotation_file = '/home/byounggun/AUE8088/utils/eval/KAIST_val-D_annotation.json'
prediction_file = '/home/byounggun/AUE8088/runs/val/exp102/best_predictions.json'

# GT 애노테이션 파일 분석
print("=== GT 애노테이션 파일 분석 ===")
with open(annotation_file, 'r') as f:
    gt_data = json.load(f)

# GT 기본 정보 출력
print(f"GT 데이터 구조 타입: {type(gt_data)}")
if isinstance(gt_data, dict):
    for key in gt_data.keys():
        print(f"키: {key}, 타입: {type(gt_data[key])}")

# GT annotations 정보 확인
if isinstance(gt_data, dict) and 'annotations' in gt_data:
    gt_annotations = gt_data['annotations']
    print(f"\nGT 전체 어노테이션 수: {len(gt_annotations)}")
    
    # 처음 5개 어노테이션만 출력
    print("\nGT 처음 5개 어노테이션:")
    for i, ann in enumerate(gt_annotations[:5]):
        print(f"\nGT 어노테이션 #{i+1}:")
        print(f"image_id: {ann.get('image_id')}")
        print(f"category_id: {ann.get('category_id')}")
        print(f"bbox: {ann.get('bbox')}") # [x, y, width, height] 형식
        print(f"기타 키: {[k for k in ann.keys() if k not in ['image_id', 'category_id', 'bbox']]}")

# 예측 결과 분석
print("\n\n=== 예측 결과 파일 분석 ===")
try:
    with open(prediction_file, 'r') as f:
        pred_data = json.load(f)
    
    print(f"예측 데이터 구조 타입: {type(pred_data)}")
    print(f"예측 결과 항목 수: {len(pred_data)}")
    
    # 처음 5개 예측만 출력
    print("\n처음 5개 예측 결과:")
    for i, pred in enumerate(pred_data[:5]):
        print(f"\n예측 #{i+1}:")
        print(f"image_id: {pred.get('image_id')}")
        print(f"image_name: {pred.get('image_name')}")
        print(f"category_id: {pred.get('category_id')}")
        print(f"bbox: {pred.get('bbox')}") # [x, y, width, height] 형식인지 확인
        print(f"score: {pred.get('score')}")
        print(f"기타 키: {[k for k in pred.keys() if k not in ['image_id', 'image_name', 'category_id', 'bbox', 'score']]}")
except Exception as e:
    print(f"예측 파일 로드 중 오류: {e}")

# 좌표계 비교 및 분석
print("\n\n=== 좌표계 및 포맷 비교 ===")
if isinstance(gt_data, dict) and 'annotations' in gt_data and 'pred_data' in locals() and len(pred_data) > 0:
    print("\n1. 바운딩 박스 형식:")
    print(f"   GT bbox: {gt_annotations[0].get('bbox')} (좌상단 xywh 형식)")
    print(f"   예측 bbox: {pred_data[0].get('bbox')} (좌상단 xywh 형식)")
    
    print("\n2. 카테고리 ID:")
    print(f"   GT: {gt_annotations[0].get('category_id')}")
    print(f"   예측: {pred_data[0].get('category_id')}")
    
    print("\n3. image_id 형식:")
    print(f"   GT: {gt_annotations[0].get('image_id')} (정수)")
    print(f"   예측: {pred_data[0].get('image_id')} (정수 또는 문자열)")
    
    print("\n4. 기타 차이점:")
    print(f"   GT 추가 필드: {[k for k in gt_annotations[0].keys() if k not in ['image_id', 'category_id', 'bbox']]}")
    print(f"   예측 추가 필드: {[k for k in pred_data[0].keys() if k not in ['image_id', 'category_id', 'bbox']]}")
