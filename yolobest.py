from pygments.formatters import img
from ultralytics import  YOLO
import cv2
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#모델사용(yolov8)
model = YOLO('./runs/detect/result7_plus/weights/best.pt')

#모델학습
results = model.train(data='data.yaml',
                      project='C:/Users/default.DESKTOP-T7K82N0/PycharmProjects/sengche/final/runs/detect',
                      name='result7_plus',
                      epochs=30,
                      imgsz=640,
                      batch=16,
                      device='cpu',
                      verbose=True,
                      save=True)

#모델검증
metrics = model.val()
