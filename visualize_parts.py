# -*- coding: utf-8 -*-
"""部件检测器可视化：在图上画 9 类部件框，人工判断检测准确度。

流程：车辆检测(YOLOv8s) -> 裁剪 -> 部件检测器(9类) -> 框映射回原图 -> 画框保存。
每类部件不同颜色 + 标签 + 置信度，车辆框（绿）一并画出。
支持单图或目录批量。

用法：
    python visualize_parts.py <图片路径>
    python visualize_parts.py <目录>        # 批量处理目录下所有图
"""
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from detect_vehicle import _get_model, MODEL_PATH, _filter_vehicle_boxes, PARTS_MODEL_PATH

# 9 类部件颜色（BGR）
PART_COLORS = {
    "taillight": (0, 0, 255), "headlight": (255, 255, 0),
    "mirror": (255, 0, 255), "window": (0, 255, 255),
    "wheel": (128, 128, 0), "plate": (0, 128, 255),
    "grille": (255, 128, 0), "bumper": (128, 0, 128),
    "exhaust": (200, 200, 0),
}
CONF = 0.25  # 部件检测置信度阈值（与 detect_taillights 一致）
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def imread_unicode(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(p, img):
    cv2.imencode(os.path.splitext(p)[1], img)[1].tofile(p)


def visualize_one(image_path):
    img = imread_unicode(image_path)
    if img is None:
        print("读图失败:", image_path)
        return
    h, w = img.shape[:2]
    res = _get_model(MODEL_PATH)(img, classes=[2, 5, 7], conf=0.4)[0]
    parts_model = _get_model(PARTS_MODEL_PATH)
    names = parts_model.names  # {0: 'taillight', ...}
    out = img.copy()

    boxes = _filter_vehicle_boxes(res.boxes, h, w)
    n_parts = 0
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 车辆框（绿）
        vc = img[y1:y2, x1:x2]
        pres = parts_model.predict(vc, conf=CONF, verbose=False)[0]
        if pres.boxes is None:
            continue
        for pb in pres.boxes:
            px1, py1, px2, py2 = pb.xyxy[0].int().tolist()
            cls = names[int(pb.cls)]
            cf = float(pb.conf)
            ox1, oy1, ox2, oy2 = x1 + px1, y1 + py1, x1 + px2, y1 + py2  # 映射回原图
            color = PART_COLORS.get(cls, (255, 255, 255))
            cv2.rectangle(out, (ox1, oy1), (ox2, oy2), color, 2)
            cv2.putText(out, f"{cls} {cf:.2f}", (ox1, oy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            n_parts += 1

    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = f"{os.path.splitext(image_path)[0]}_parts"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{stem}_parts.jpg")
    imwrite_unicode(out_path, out)
    print(f"{os.path.basename(image_path)}: 车辆 {len(boxes)} 部件 {n_parts} -> {out_path}")


def main():
    if len(sys.argv) < 2:
        print("用法: python visualize_parts.py <图片或目录>")
        sys.exit(1)
    target = sys.argv[1]
    if os.path.isdir(target):
        imgs = [p for p in glob.glob(os.path.join(target, "*")) if p.lower().endswith(IMG_EXTS)]
        print(f"目录 {target}: {len(imgs)} 张图")
        for p in imgs:
            visualize_one(p)
    else:
        visualize_one(target)


if __name__ == "__main__":
    main()
