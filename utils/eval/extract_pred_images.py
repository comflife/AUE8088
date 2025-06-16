#!/usr/bin/env python3
import json
import os
import argparse

def extract_image_list(pred_file, output_file):
    # 예측 파일 로드
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    # 중복 없이 이미지 이름 추출
    image_names = set()
    for pred in predictions:
        if 'image_name' in pred:
            # .jpg 확장자 추가 (KAIST 형식과 일치)
            img_name = pred['image_name'] + '.jpg'
            image_names.add(img_name)
    
    # 정렬된 목록으로 변환
    sorted_names = sorted(image_names)
    
    # 텍스트 파일에 저장
    with open(output_file, 'w') as f:
        for name in sorted_names:
            f.write(f"{name}\n")
    
    print(f"추출된 이미지 수: {len(sorted_names)}")
    print(f"이미지 목록이 {output_file}에 저장되었습니다.")
    
    # 이미지 세트 분석
    sets = {}
    for name in sorted_names:
        # KAIST 형식: set00_V000_I00019.jpg
        parts = name.split('_')
        if len(parts) > 0:
            set_name = parts[0]
            if set_name in sets:
                sets[set_name] += 1
            else:
                sets[set_name] = 1
    
    print("\n이미지 세트 분포:")
    for set_name, count in sets.items():
        print(f"{set_name}: {count}개 이미지")

def parse_args():
    parser = argparse.ArgumentParser(description="예측 파일에서 이미지 목록 추출")
    parser.add_argument('--pred_file', default='/home/byounggun/AUE8088/runs/val/exp71/best_predictions.json', 
                        help='예측 JSON 파일 경로')
    parser.add_argument('--output', default='/home/byounggun/AUE8088/kaist-rgbt/pred-images.txt', 
                        help='출력 텍스트 파일 경로')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    extract_image_list(args.pred_file, args.output)
