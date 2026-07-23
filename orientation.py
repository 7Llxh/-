# -*- coding: utf-8 -*-
"""车辆朝向识别（F3）。YOLOv8n-cls 三分类 front/rear/side。

推理接口 detect_orientation，供 check_taillight_detection 筛 rear 用
（只检查 rear 图的尾灯漏检，不再分流到整车库）。

详见 朝向识别-技术设计.md。
"""
from ultralytics import YOLO

ORIENTATION_MODEL_PATH = "runs/cls/orientation/weights/best.pt"
# ImageFolder 按字母序：front < rear < side，与训练时类别顺序一致
CLASSES = ["front", "rear", "side"]
CONF_THRESHOLD = 0.6  # 低于此值输出 uncertain

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = YOLO(ORIENTATION_MODEL_PATH)
    return _model


def detect_orientation(vehicle_crop, conf_threshold=CONF_THRESHOLD):
    """识别单辆车的朝向。

    参数:
        vehicle_crop: 车辆裁剪图（BGR ndarray，非定尺寸）
        conf_threshold: 低于此置信度输出 uncertain
    返回:
        {"view": front|rear|side|uncertain, "confidence": float, "raw_probs": {...}}
    """
    res = _get_model().predict(vehicle_crop, imgsz=224, verbose=False)[0]
    probs = res.probs
    top = int(probs.top1)
    conf = float(probs.top1conf)
    view = CLASSES[top] if conf >= conf_threshold else "uncertain"
    return {
        "view": view,
        "confidence": conf,
        "raw_probs": {CLASSES[i]: float(probs.data[i]) for i in range(len(CLASSES))},
    }


if __name__ == "__main__":
    import sys
    import cv2
    import numpy as np
    from detect_vehicle import _get_model as gvm, MODEL_PATH, _filter_vehicle_boxes

    path = sys.argv[1] if len(sys.argv) > 1 else input("图片路径: ").strip().strip('"')
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("读图失败:", path); sys.exit(1)
    h, w = img.shape[:2]
    r = gvm(MODEL_PATH)(img, classes=[2, 5, 7], conf=0.4)[0]
    for box in _filter_vehicle_boxes(r.boxes, h, w):
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        ori = detect_orientation(img[y1:y2, x1:x2])
        print(f"[{x1},{y1},{x2},{y2}] {ori['view']} (conf={ori['confidence']:.2f}) "
              f"{ {k: round(v,2) for k,v in ori['raw_probs'].items()} }")
