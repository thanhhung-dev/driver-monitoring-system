"""
Phân tích logs/headpose_trace.csv → chẩn đoán nguyên nhân flip yaw real-time.

CHẠY:
    cd D:\\Workspace\\driver_monitoring\\dms
    python -m test.analyze_headpose_trace [path_csv]

Mặc định đọc logs/headpose_trace.csv. In ra:
  1. Tổng quan: tổng frame, số frame per branch
  2. Các event flip dấu (raw_yaw nhảy qua 0 giữa 2 frame gần nhau)
  3. Chẩn đoán nguyên nhân dựa trên R/sy/det tại các event flip
  4. Top 10 frame có sy thấp nhất (gần gimbal lock)
"""

import csv
import os
import sys
from collections import Counter

import numpy as np


def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _f(x, default=float("nan")):
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "logs", "headpose_trace.csv")
    if not os.path.exists(path):
        print(f"[ERR] Không thấy CSV: {path}")
        print("      Hãy chạy: set DMS_HEADPOSE_DEBUG=1 && python main.py")
        sys.exit(1)

    rows = load_csv(path)
    if not rows:
        print("[ERR] CSV rỗng.")
        sys.exit(1)

    print(f"\n=== Phân tích {path} ===")
    print(f"Tổng số dòng (frame): {len(rows)}")

    # 1) Phân bố nhánh
    branches = Counter(r["branch"] for r in rows)
    print("\n--- Phân bố nhánh (branch) ---")
    for b, n in branches.most_common():
        print(f"  {b:25s} : {n}")

    # 2) Chỉ lọc các dòng có dữ liệu model (bỏ skip/no_bbox)
    valid = [r for r in rows if r["branch"] not in ("", "no_bbox", "skip_interval")]
    if not valid:
        print("\n[!] Không có dòng nào chạy model. Kiểm tra lại pipeline.")
        return

    print(f"\n--- Frames chạy model: {len(valid)} ---")

    # 3) Tìm các event FLIP dấu raw_yaw (qua 0 giữa 2 frame gần nhau, |yaw| lớn)
    print("\n--- Event FLIP dấu raw_yaw (giữa 2 frame liên tiếp) ---")
    flips = []
    for i in range(1, len(valid)):
        y_prev = _f(valid[i - 1]["raw_yaw"])
        y_cur = _f(valid[i]["raw_yaw"])
        if np.isnan(y_prev) or np.isnan(y_cur):
            continue
        # Flip = đổi dấu VÀ cả 2 đều có độ lớn đáng kể
        if y_prev * y_cur < 0 and abs(y_prev) > 40 and abs(y_cur) > 40:
            flips.append((i, valid[i - 1], valid[i]))

    if not flips:
        print("  (không có event flip raw_yaw nào thỏa điều kiện)")
    else:
        print(f"  Tìm thấy {len(flips)} event flip:\n")
        for idx, (i, prev, cur) in enumerate(flips):
            print(f"  #{idx + 1} frame {prev['frame']} → {cur['frame']}")
            print(f"      raw_yaw:   {prev['raw_yaw']:>7} → {cur['raw_yaw']:>7}")
            print(f"      branch:    {prev['branch']:>15} → {cur['branch']:>15}")
            print(f"      kp_sign:   {prev['kp_sign']:>7} → {cur['kp_sign']:>7}")
            print(f"      ‖RᵀR−I‖:  {prev['ortho_err']:>7} → {cur['ortho_err']:>7}")
            print(f"      det_R:     {prev['det_R']:>7} → {cur['det_R']:>7}")
            print(f"      sy:        {prev['sy']:>7} → {cur['sy']:>7}   (sy<0.1 = gần gimbal lock)")
            print(f"      extreme:   {prev['extreme']:>7} → {cur['extreme']:>7}")
            print(f"      out_yaw:   {prev['out_yaw']:>7} → {cur['out_yaw']:>7}")
            print()

    # 4) Chẩn đoán nguyên nhân dựa trên các chỉ số R tại event flip
    print("--- Chẩn đoán nguyên nhân tại các event flip ---")
    if not flips:
        print("  (không có event flip để chẩn đoán)")
    else:
        for idx, (i, prev, cur) in enumerate(flips):
            notes = []
            for label, r in [("frame_trước", prev), ("frame_sau", cur)]:
                ortho = _f(r["ortho_err"])
                detR = _f(r["det_R"])
                sy = _f(r["sy"])
                if detR < 0:
                    notes.append(f"  {label}: det(R)={detR:+.3f} < 0 → REFLECTION")
                if ortho > 0.01:
                    notes.append(f"  {label}: ‖RᵀR−I‖={ortho:.4f} > 0.01 → R hỏng (model OOD)")
                if sy < 0.1:
                    notes.append(f"  {label}: sy={sy:.4f} < 0.1 → gần GIMBAL LOCK")
                if ortho < 0.01 and detR > 0 and sy >= 0.1:
                    notes.append(f"  {label}: R hợp lệ + sy ok → flip do công thức/ambiguity 2D")
            print(f"  Flip #{idx + 1}:")
            for n in notes:
                print(n)
            print()

    # 5) Top 10 frame có sy thấp nhất (gần gimbal lock nhất)
    print("--- Top 10 frame có sy thấp nhất (gần gimbal lock) ---")
    sy_sorted = sorted(valid, key=lambda r: _f(r["sy"]))[:10]
    for r in sy_sorted:
        print(f"  frame {r['frame']:>5} | sy={r['sy']:>7} | "
              f"raw_yaw={r['raw_yaw']:>7} | ortho={r['ortho_err']:>7} | det={r['det_R']:>7}")

    # 6) Kiểm tra tỉ lệ R hỏng toàn trace
    print("\n--- Tỉ lệ R hỏng trên toàn trace ---")
    bad_R = sum(1 for r in valid if _f(r["ortho_err"]) > 0.01 or _f(r["det_R"]) < 0)
    print(f"  R hỏng: {bad_R}/{len(valid)} = {100 * bad_R / len(valid):.1f}%")
    if bad_R / len(valid) > 0.05:
        print("  → R hay hỏng (>5%): nghi ngờ model OOD hoặc output layer sai.")


if __name__ == "__main__":
    main()
