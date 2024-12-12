from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# YOLOv8 모델 로드
# 사전 학습된 모델 사용 ('yolov8n.pt'는 nano 모델, 더 성능 좋은 모델은 'yolov8s.pt', 'yolov8m.pt' 등 사용 가능)
model = YOLO('./runs/detect/result7_plus/weights/best.pt')

# YOLO 감지 함수
def yolo_detect_and_visualize(model, img_path):
    # 이미지 로드
    img_array = np.fromfile(img_path, np.uint8)
    if img_array is None:
        print(f"이미지를 로드할 수 없습니다: {img_path}")
        return
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # YOLO 모델로 예측 수행
    results = model(image)

    # 결과 시각화 및 출력
    result_image = results[0].plot()  # YOLO 감지 결과를 시각화한 이미지
    detected_objects = results[0].boxes.data.cpu().numpy()  # 감지된 객체 정보 (좌표, 클래스, 확률)

    # 감지된 객체 정보 출력
    for obj in detected_objects:
        x1, y1, x2, y2, confidence, class_id = obj
        print(f"Class: {class_id}, Confidence: {confidence:.2f}, Box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    # 감지 결과 이미지 출력
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title("YOLO Detection Results")
    plt.axis("off")
    plt.show()

# 감지 실행
sample_image_path = "test3.jpg"  # 예측에 사용할 이미지 경로
yolo_detect_and_visualize(model, sample_image_path)
