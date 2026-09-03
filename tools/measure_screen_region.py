#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import cv2
import mss
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="框选屏幕区域并输出 left/top/width/height"
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=0,
        help="mss 显示器编号：0=整个虚拟屏幕，1=第一块显示器",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with mss.MSS() as sct:
        if args.monitor >= len(sct.monitors):
            raise SystemExit(f"显示器编号不存在: {args.monitor}")

        monitor = sct.monitors[args.monitor]
        if monitor["width"] <= 0 or monitor["height"] <= 0:
            raise SystemExit(
                "无法读取屏幕信息。请在系统设置中给终端/Python 授予屏幕录制权限。"
            )

        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    scale_x = frame.shape[1] / monitor["width"]
    scale_y = frame.shape[0] / monitor["height"]

    print("在预览窗口拖选目标区域，然后按 Enter 或 Space 确认；按 Esc 取消。")
    x, y, width, height = cv2.selectROI(
        "Select monitoring region", frame, showCrosshair=True
    )
    cv2.destroyAllWindows()

    if width == 0 or height == 0:
        raise SystemExit("未选择区域。")

    # Retina 屏幕的截图像素可能是系统坐标的两倍，这里换算回 mss 坐标。
    left = int(round(monitor["left"] + x / scale_x))
    top = int(round(monitor["top"] + y / scale_y))
    width = int(round(width / scale_x))
    height = int(round(height / scale_y))

    print("\n屏幕区域:")
    print(f"left: {left}")
    print(f"top: {top}")
    print(f"width: {width}")
    print(f"height: {height}")

    print("\nconfig.yaml 片段:")
    print("window:")
    print("  default:")
    print(f"    left: {left}")
    print(f"    top: {top}")
    print(f"    width: {width}")
    print(f"    height: {height}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
