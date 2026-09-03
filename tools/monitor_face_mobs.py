#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import cv2

from auto import OptimizedMapleBot


TARGET_CLASSES = {"木面怪人", "石面怪人"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="監控螢幕指定區域，只輸出木面怪人/石面怪人偵測結果"
    )
    parser.add_argument("--interval", type=float, default=0.5, help="偵測間隔秒數")
    parser.add_argument(
        "--save-dir",
        default="validation/monitor",
        help="發現目標時保存標註圖的目錄",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bot = OptimizedMapleBot()
    if bot.model is None:
        raise SystemExit("模型載入失敗")

    monitor = bot.monitor
    print(
        f"監控區域: left={monitor['left']}, top={monitor['top']}, "
        f"width={monitor['width']}, height={monitor['height']}"
    )
    print("按 Ctrl+C 停止。")

    frame_no = 0
    save_dir = Path(args.save_dir)
    try:
        while True:
            frame = bot.capture_screen()
            if frame is None:
                time.sleep(args.interval)
                continue

            detections = [
                item
                for item in bot.detect_objects(frame)
                if item.class_name in TARGET_CLASSES
            ]
            timestamp = time.strftime("%H:%M:%S")
            if detections:
                print(f"[{timestamp}] 發現 {len(detections)} 個目標")
                for detection in detections:
                    print(
                        f"  {detection.class_name} conf={detection.confidence:.2f} "
                        f"center={detection.center}"
                    )

                frame_no += 1
                output = bot._draw_detections(frame, detections)
                save_dir.mkdir(parents=True, exist_ok=True)
                path = (
                    save_dir
                    / f"target_{time.strftime('%Y%m%d_%H%M%S')}_{frame_no}.jpg"
                )
                cv2.imwrite(str(path), output)
                print(f"  已保存: {path}")
            else:
                print(f"[{timestamp}] 未發現目標")

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("監控已停止。")


if __name__ == "__main__":
    main()
