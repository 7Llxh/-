# -*- coding: utf-8 -*-
"""批量评估端到端识别准确率。

跑尾灯库内系列的图，统计 top1 准确率、未识别尾灯率、per-series 效果。
能看出是 embedder 问题还是其他环节（尾灯漏检等）。

用法:
    python eval_recognize.py           # 每系列测试 20 张
    python eval_recognize.py --n 50    # 每系列 50 张
"""
import argparse
import contextlib
import io
import json
import os
import sys
import glob
import random

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from recognize import recognize

TL_META = os.path.join(HERE, "data", "features", "taillight_meta.json")
RAW = os.path.join(HERE, "data", "raw")
EXTS = (".jpg", ".jpeg", ".png", ".webp")


def box_area(r):
    b = r.get("box", [0, 0, 0, 0])
    return (b[2] - b[0]) * (b[3] - b[1])


def main():
    ap = argparse.ArgumentParser(description="批量评估端到端识别准确率")
    ap.add_argument("--n", type=int, default=20, help="每系列测试图数（默认20）")
    args = ap.parse_args()

    meta = json.load(open(TL_META, encoding="utf-8"))
    lib_series = sorted(set(m["model_series"] for m in meta))
    print(f"尾灯库系列({len(lib_series)}): {lib_series}")
    print(f"每系列测试 {args.n} 张图\n")

    random.seed(42)
    stats = {}
    total = {"imgs": 0, "correct": 0, "no_tail": 0, "wrong": 0}
    details = []

    for series in lib_series:
        # 找该系列图（data/raw/{series}_*/*.jpg）
        imgs = []
        for d in sorted(glob.glob(os.path.join(RAW, series + "_*"))):
            if os.path.isdir(d):
                imgs += [p for p in glob.glob(os.path.join(d, "*"))
                         if p.lower().endswith(EXTS)]
        if not imgs:
            continue
        random.shuffle(imgs)
        imgs = imgs[:args.n]

        s = {"total": 0, "correct": 0, "no_tail": 0, "wrong": 0}
        for img_path in imgs:
            # 抑制 recognize 的 print（只取返回值）
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                results = recognize(img_path)
            if not results:
                continue
            # 取最大框（主车）
            main_car = max(results, key=box_area)
            s["total"] += 1
            total["imgs"] += 1
            fname = os.path.basename(img_path)
            if "result" in main_car:  # 未识别到车尾灯
                s["no_tail"] += 1
                total["no_tail"] += 1
                details.append({"file": fname, "true": series, "pred": "", "status": "未识别尾灯"})
            elif main_car.get("make_model") == series:
                s["correct"] += 1
                total["correct"] += 1
                details.append({"file": fname, "true": series, "pred": main_car["make_model"], "status": "正确"})
            else:
                s["wrong"] += 1
                total["wrong"] += 1
                details.append({"file": fname, "true": series, "pred": main_car.get("make_model", "?"), "status": "误识别"})
        stats[series] = s
        acc = s["correct"] / s["total"] if s["total"] else 0
        print(f"  {series}: {s['total']}图 正确{s['correct']} 未识别尾灯{s['no_tail']} 误识别{s['wrong']} 准确率{acc:.0%}", flush=True)

    # 汇总
    print(f"\n{'='*75}")
    print(f"{'系列':<25} {'测试':<6} {'正确':<6} {'未识别尾灯':<10} {'误识别':<6} {'准确率'}")
    print("-" * 75)
    for series, s in stats.items():
        acc = s["correct"] / s["total"] if s["total"] else 0
        print(f"{series:<25} {s['total']:<6} {s['correct']:<6} {s['no_tail']:<10} {s['wrong']:<6} {acc:.1%}")
    print("-" * 75)
    acc = total["correct"] / total["imgs"] if total["imgs"] else 0
    no_tail_rate = total["no_tail"] / total["imgs"] if total["imgs"] else 0
    wrong_rate = total["wrong"] / total["imgs"] if total["imgs"] else 0
    print(f"{'总计':<25} {total['imgs']:<6} {total['correct']:<6} {total['no_tail']:<10} {total['wrong']:<6} {acc:.1%}")
    print(f"\n端到端 top1 准确率: {acc:.1%} ({total['correct']}/{total['imgs']})")
    print(f"未识别尾灯率: {no_tail_rate:.1%} ({total['no_tail']}/{total['imgs']})")
    print(f"误识别率: {wrong_rate:.1%} ({total['wrong']}/{total['imgs']})")

    # 输出每图明细 CSV
    import csv
    csv_path = os.path.join(HERE, "eval_results.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "true", "pred", "status"])
        w.writeheader()
        w.writerows(details)
    print(f"\n明细已输出: {csv_path}")


if __name__ == "__main__":
    main()
