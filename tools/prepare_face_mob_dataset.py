#!/usr/bin/env python3
"""Build a synthetic YOLO dataset for the two masked mobs."""

import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


SPRITES = [
    (0, "木面怪人", "datasets/reference/mob_page1/2230110_木面怪人.png"),
    (1, "石面怪人", "datasets/reference/mob_page1/2230111_石面怪人.png"),
]
NEGATIVE_DIR = Path("datasets/reference/mob_page1")


def make_background(size: int, rng: random.Random) -> Image.Image:
    top = tuple(rng.randint(45, 235) for _ in range(3))
    bottom = tuple(max(0, min(255, value + rng.randint(-75, 75))) for value in top)
    image = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(image)

    for y in range(size):
        ratio = y / (size - 1)
        color = tuple(int(top[i] * (1 - ratio) + bottom[i] * ratio) for i in range(3))
        draw.line([(0, y), (size, y)], fill=color)

    for _ in range(rng.randint(3, 12)):
        x0 = rng.randint(-size // 4, size)
        y0 = rng.randint(-size // 4, size)
        x1 = x0 + rng.randint(20, size // 2)
        y1 = y0 + rng.randint(10, size // 3)
        tone = rng.randint(35, 75)
        color = tuple(max(0, min(255, channel + rng.randint(-tone, tone))) for channel in top)
        draw.rectangle((x0, y0, x1, y1), fill=color)

    return image


def prepare_sprite(sprite: Image.Image, rng: random.Random) -> Image.Image:
    scale = rng.uniform(0.38, 1.25)
    width = max(16, int(sprite.width * scale))
    height = max(16, int(sprite.height * scale))
    sprite = sprite.resize((width, height), Image.Resampling.BILINEAR)

    if rng.random() < 0.5:
        sprite = sprite.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.65:
        sprite = sprite.rotate(rng.uniform(-14, 14), expand=True, resample=Image.Resampling.BILINEAR)

    sprite = ImageEnhance.Brightness(sprite).enhance(rng.uniform(0.78, 1.22))
    sprite = ImageEnhance.Contrast(sprite).enhance(rng.uniform(0.82, 1.20))
    sprite = ImageEnhance.Color(sprite).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.35:
        sprite = sprite.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.9)))
    if rng.random() < 0.25:
        alpha = sprite.getchannel("A").point(lambda value: 255 if value > 160 else 0)
        sprite.putalpha(alpha)
    return sprite


def paste_instance(canvas: Image.Image, sprite: Image.Image, rng: random.Random) -> tuple[int, int, int, int] | None:
    max_x = canvas.width - sprite.width - 1
    max_y = canvas.height - sprite.height - 1
    if max_x < 1 or max_y < 1:
        return None

    x = rng.randint(1, max_x)
    y = rng.randint(1, max_y)
    if rng.random() < 0.3:
        sprite = sprite.copy()
        alpha = sprite.getchannel("A").point(lambda value: int(value * rng.uniform(0.72, 0.98)))
        sprite.putalpha(alpha)

    canvas.paste(sprite, (x, y), sprite)
    alpha_bbox = sprite.getchannel("A").point(lambda value: 255 if value > 24 else 0).getbbox()
    if not alpha_bbox:
        return None

    left, top, right, bottom = alpha_bbox
    return (x + left, y + top, x + right, y + bottom)


def finish_background(canvas: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 0.7)))
    if rng.random() < 0.35:
        canvas = ImageEnhance.Brightness(canvas).enhance(rng.uniform(0.86, 1.14))
    if rng.random() < 0.35:
        canvas = ImageEnhance.Contrast(canvas).enhance(rng.uniform(0.88, 1.16))
    return canvas


def make_sample(
    size: int,
    sprites: list[tuple[int, str, Image.Image]],
    negatives: list[Image.Image],
    rng: random.Random,
) -> tuple[Image.Image, list[tuple[int, int, int, int, int]]]:
    canvas = make_background(size, rng)
    labels = []
    for _ in range(rng.randint(0, 5)):
        if not negatives:
            break
        negative = prepare_sprite(rng.choice(negatives), rng)
        paste_instance(canvas, negative, rng)

    count = rng.choices((0, 1, 2, 3), weights=(0.12, 0.38, 0.32, 0.18))[0]
    class_ids = [rng.randint(0, len(sprites) - 1) for _ in range(count)]

    for class_id in class_ids:
        _, _, base_sprite = sprites[class_id]
        sprite = prepare_sprite(base_sprite, rng)
        bbox = paste_instance(canvas, sprite, rng)
        if bbox:
            labels.append((class_id, *bbox))

    return finish_background(canvas, rng), labels


def write_split(
    split: str,
    count: int,
    size: int,
    sprites: list[tuple[int, str, Image.Image]],
    negatives: list[Image.Image],
    output: Path,
    rng: random.Random,
) -> list[int]:
    images_dir = output / "images" / split
    labels_dir = output / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    class_counts = [0] * len(sprites)

    for index in range(count):
        image, boxes = make_sample(size, sprites, negatives, rng)
        image_path = images_dir / f"{split}_{index:05d}.png"
        image.save(image_path)

        lines = []
        for class_id, left, top, right, bottom in boxes:
            x_center = (left + right) / 2 / size
            y_center = (top + bottom) / 2 / size
            width = (right - left) / size
            height = (bottom - top) / size
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            class_counts[class_id] += 1

        (labels_dir / f"{split}_{index:05d}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return class_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("datasets/synthetic_face_mobs"))
    parser.add_argument("--train-count", type=int, default=240)
    parser.add_argument("--val-count", type=int, default=40)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sprites = [(class_id, name, Image.open(path).convert("RGBA")) for class_id, name, path in SPRITES]
    target_paths = {Path(path).resolve() for _, _, path in SPRITES}
    negatives = [
        Image.open(path).convert("RGBA")
        for path in sorted(NEGATIVE_DIR.glob("*.png"))
        if path.resolve() not in target_paths
    ]
    train_counts = write_split("train", args.train_count, args.size, sprites, negatives, args.output, rng)
    val_counts = write_split("val", args.val_count, args.size, sprites, negatives, args.output, rng)

    data_yaml = args.output / "data.yaml"
    names = "\n".join(f"  {class_id}: {name}" for class_id, name, _ in sprites)
    data_yaml.write_text(
        f"path: {args.output.resolve()}\ntrain: images/train\nval: images/val\n\nnames:\n{names}\n",
        encoding="utf-8",
    )
    summary = {
        "classes": {str(class_id): name for class_id, name, _ in sprites},
        "train_images": args.train_count,
        "val_images": args.val_count,
        "hard_negatives": len(negatives),
        "train_instances": dict(zip([name for _, name, _ in sprites], train_counts)),
        "val_instances": dict(zip([name for _, name, _ in sprites], val_counts)),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
