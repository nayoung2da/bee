import os
import shutil
import json
from sklearn.model_selection import train_test_split


def create_yolo_labels(image_root, label_root, output_dir, split_ratio=0.8):
    image_paths = []
    label_paths = []

    # 모든 이미지와 라벨 파일 경로 수집
    for root, dirs, files in os.walk(image_root):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(root, file)
                label_path = img_path.replace(image_root, label_root).replace('.jpg', '.json').replace('.png', '.json')

                if os.path.exists(label_path):
                    image_paths.append(img_path)
                    label_paths.append(label_path)

    # 학습용과 검증용으로 분할
    train_images, val_images, train_labels, val_labels = train_test_split(image_paths, label_paths,
                                                                          train_size=split_ratio, random_state=42)

    def process_split(image_list, label_list, split_name):
        image_output_dir = os.path.join(output_dir, split_name, 'images')
        label_output_dir = os.path.join(output_dir, split_name, 'labels')
        os.makedirs(image_output_dir, exist_ok=True)
        os.makedirs(label_output_dir, exist_ok=True)

        for img_path, label_path in zip(image_list, label_list):
            # 이미지 파일 복사
            shutil.copy2(img_path, image_output_dir)

            # JSON 파일 읽기
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {label_path}: {e}")
                continue

            # YOLO 형식의 라벨 파일 생성
            try:
                image_width = data['image']['width']
                image_height = data['image']['height']
                yolo_label_path = os.path.join(label_output_dir,
                                               os.path.basename(img_path).replace('.jpg', '.txt').replace('.png',
                                                                                                          '.txt'))

                with open(yolo_label_path, 'w') as label_file:
                    for ann in data.get('annotations', []):
                        class_id = ann.get('category_id', -1)  # JSON에서의 class_id
                        bbox = ann.get('bbox', [])  # 확인해야 할 bbox 형식

                        if len(bbox) != 4 or class_id == -1:
                            print(f"Invalid annotation in {label_path}: {ann}")
                            continue

                        # 바운딩 박스 좌표 변환
                        if isinstance(bbox, list):
                            if bbox[2] > 1 and bbox[3] > 1:  # x_max, y_max 형식일 때
                                x_min, y_min, x_max, y_max = bbox
                                bbox_width = x_max - x_min
                                bbox_height = y_max - y_min
                            else:  # width, height 형식일 때
                                x_min, y_min, bbox_width, bbox_height = bbox
                        else:
                            print(f"Invalid bbox format in {label_path}: {ann}")
                            continue

                        # 중심 좌표 및 크기 계산
                        x_center = (x_min + bbox_width / 2) / image_width
                        y_center = (y_min + bbox_height / 2) / image_height
                        width = bbox_width / image_width
                        height = bbox_height / image_height

                        # 좌표 값들이 0과 1 사이인지 확인
                        if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
                            label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        else:
                            print(f"WARNING: Bbox out of bounds in image {img_path} with bbox {bbox}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    # 학습 데이터와 검증 데이터 처리
    process_split(train_images, train_labels, 'train')
    process_split(val_images, val_labels, 'val')


# 경로 설정
image_root = 'C:/Users/default.DESKTOP-T7K82N0/PycharmProjects/sengche/final/Sample/01.원천데이터'
label_root = 'C:/Users/default.DESKTOP-T7K82N0/PycharmProjects/sengche/final/Sample/02.라벨링데이터'
output_dir = 'C:/Users/default.DESKTOP-T7K82N0/PycharmProjects/sengche/final/Sample/output_yolo_dataset'

# 데이터셋 준비
create_yolo_labels(image_root, label_root, output_dir)