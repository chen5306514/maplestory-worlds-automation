#!/usr/bin/env python3
import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


CLASS_NAMES = {0: "player_right", 1: "player_left"}


def yolo_line(class_id: int, x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> str:
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def prepare_dataset(source_dir: Path, output_dir: Path) -> int:
    labels_path = source_dir / "labels.csv"
    images_path = source_dir / "images"
    if not labels_path.exists() or not images_path.exists():
        raise FileNotFoundError("需要 datasets/player_real/labels.csv 和 datasets/player_real/images")

    rows = list(csv.DictReader(labels_path.open(encoding="utf-8")))
    if not rows:
        raise ValueError("labels.csv 没有任何标注")

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    by_facing: dict[str, list[dict]] = {"right": [], "left": []}
    for row in rows:
        facing = row["facing"].strip().lower()
        if facing not in by_facing:
            continue
        by_facing[facing].append(row)

    rng = random.Random(42)
    split_rows: dict[str, list[tuple[dict, bool]]] = {"train": [], "val": []}
    for facing in ("right", "left"):
        items = by_facing[facing]
        rng.shuffle(items)
        val_count = 1 if len(items) > 1 else 0
        for row in items[:val_count]:
            split_rows["val"].append((row, False))
        for row in items[val_count:]:
            split_rows["train"].append((row, False))

    # 左向样本由右向原图水平镜像生成；YOLO 训练时必须关闭 fliplr。
    for row in rows:
        if row["facing"].strip().lower() == "right":
            split_rows["train"].append((row, True))
            split_rows["val"].append((row, True))

    count = 0
    for split, samples in split_rows.items():
        for row, mirrored in samples:
            source = images_path / row["image"]
            image = cv2.imread(str(source))
            if image is None:
                continue

            height, width = image.shape[:2]
            x1, y1, x2, y2 = map(int, (row["x1"], row["y1"], row["x2"], row["y2"]))
            actual_facing = "left" if mirrored else row["facing"].strip().lower()
            class_id = 0 if actual_facing == "right" else 1
            if mirrored:
                image = cv2.flip(image, 1)
                x1, x2 = width - x2, width - x1
                stem = f"{source.stem}_left"
            else:
                stem = source.stem

            image_name = f"{stem}.png"
            label_name = f"{stem}.txt"
            cv2.imwrite(str(output_dir / "images" / split / image_name), image)
            (output_dir / "labels" / split / label_name).write_text(
                yolo_line(class_id, x1, y1, x2, y2, width, height) + "\n",
                encoding="utf-8",
            )
            count += 1

    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n  0: {CLASS_NAMES[0]}\n  1: {CLASS_NAMES[1]}\n",
        encoding="utf-8",
    )
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", default="weights/player.pt")
    parser.add_argument("--dataset", default="datasets/player_detection")
    parser.add_argument("--source", default="datasets/player_real")
    args = parser.parse_args()

    source_dir = Path(args.source)
    dataset_dir = Path(args.dataset)
    sample_count = prepare_dataset(source_dir, dataset_dir)
    print(f"准备完成: {sample_count} 张训练样本 ({dataset_dir})")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"训练设备: {device}")
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(dataset_dir / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=4,
        workers=2,
        device=device,
        project="runs/player",
        name="train",
        exist_ok=True,
        patience=20,
        # 类别本身表示朝向，镜像必须由数据集显式生成，不能让 YOLO 随机翻转。
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        mosaic=0.4,
        scale=0.2,
        translate=0.1,
    )

    save_dir = getattr(model.trainer, "save_dir", None)
    best = Path(save_dir) / "weights" / "best.pt" if save_dir else Path("runs/detect/runs/player/train/weights/best.pt")
    if best.exists():
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, output)
        print(f"已导出模型: {output}")


if __name__ == "__main__":
    main()
