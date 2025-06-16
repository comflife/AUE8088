#!/usr/bin/env python3
import json
import os
from pathlib import Path

# 파일 경로
annotation_file = '/home/byounggun/AUE8088/utils/eval/KAIST_val-D_annotation.json'

# GT 애노테이션 파일 분석
print("=== GT 이미지 ID 매핑 분석 ===")
with open(annotation_file, 'r') as f:
    gt_data = json.load(f)

# 이미지 ID와 파일 이름 매핑 확인
if 'images' in gt_data:
    images = gt_data['images']
    print(f"전체 이미지 수: {len(images)}")
    
    # 처음 10개 이미지의 ID 매핑
    print("\n처음 10개 이미지의 ID 매핑:")
    for i, img in enumerate(images[:10]):
        print(f"이미지 #{i+1}:")
        print(f"  ID: {img.get('id')}")
        print(f"  파일 이름: {img.get('file_name')}")
        
        # 파일 이름에서 숫자 추출 시도
        file_name = img.get('file_name')
        if file_name:
            parts = file_name.split('_')
            if len(parts) >= 3:
                try:
                    # set00_V000_I00019 형식에서 마지막 I00019 부분에서 숫자만 추출
                    num_part = parts[-1]
                    if num_part.startswith('I'):
                        img_num = int(num_part[1:])  # I00019 -> 19
                        print(f"  추출한 숫자: {img_num}")
                except:
                    print(f"  숫자 추출 실패")

    # 이미지 ID와 애노테이션 ID 매핑 관계
    print("\n이미지 ID와 애노테이션 매핑:")
    image_anno_map = {}
    for anno in gt_data.get('annotations', []):
        img_id = anno.get('image_id')
        if img_id not in image_anno_map:
            image_anno_map[img_id] = []
        image_anno_map[img_id].append(anno.get('id'))
    
    print(f"이미지 ID 종류: {sorted(image_anno_map.keys())[:10]}... (총 {len(image_anno_map)} 개)")
    
    # 첫 5개 이미지 ID에 대한 애노테이션
    for i, img_id in enumerate(sorted(image_anno_map.keys())[:5]):
        print(f"이미지 ID {img_id}:")
        print(f"  애노테이션 수: {len(image_anno_map[img_id])}")
        if i == 0:  # 첫 번째 이미지만 애노테이션 상세 표시
            for anno_id in image_anno_map[img_id][:3]:  # 최대 3개만
                for anno in gt_data.get('annotations', []):
                    if anno.get('id') == anno_id:
                        print(f"    애노테이션: {anno}")
                        break
