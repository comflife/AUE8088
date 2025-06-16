#!/usr/bin/env python3
import json
import os

# KAIST 어노테이션 파일 경로
annotation_file = '/home/byounggun/AUE8088/utils/eval/KAIST_val-D_annotation.json'

# JSON 파일 로드
with open(annotation_file, 'r') as f:
    data = json.load(f)

# 기본 정보 출력
print(f"데이터 구조 타입: {type(data)}")
if isinstance(data, dict):
    for key in data.keys():
        print(f"키: {key}, 타입: {type(data[key])}")

# images, categories, annotations 정보가 있는지 확인
if isinstance(data, dict) and 'annotations' in data:
    annotations = data['annotations']
    print(f"\n전체 어노테이션 수: {len(annotations)}")
    
    # 처음 5개 어노테이션만 출력
    print("\n처음 5개 어노테이션:")
    for i, ann in enumerate(annotations[:5]):
        print(f"\n어노테이션 #{i+1}:")
        print(f"image_id: {ann.get('image_id')}")
        print(f"category_id: {ann.get('category_id')}")
        print(f"bbox: {ann.get('bbox')}") # [x, y, width, height]
        print(f"기타 키: {[k for k in ann.keys() if k not in ['image_id', 'category_id', 'bbox']]}")

# 이미지 정보가 있으면 출력
if isinstance(data, dict) and 'images' in data:
    images = data['images']
    print(f"\n전체 이미지 수: {len(images)}")
    
    # 처음 3개 이미지만 출력
    print("\n처음 3개 이미지:")
    for i, img in enumerate(images[:3]):
        print(f"\n이미지 #{i+1}:")
        print(f"id: {img.get('id')}")
        print(f"file_name: {img.get('file_name')}")
        print(f"width: {img.get('width')}, height: {img.get('height')}")

# 카테고리 정보가 있으면 출력
if isinstance(data, dict) and 'categories' in data:
    categories = data['categories']
    print(f"\n카테고리 정보:")
    for cat in categories:
        print(f"id: {cat.get('id')}, name: {cat.get('name')}")
