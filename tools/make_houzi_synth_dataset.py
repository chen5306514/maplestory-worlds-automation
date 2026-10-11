#!/usr/bin/env python3
"""为两种猴子素材生成统一类别的 YOLO 检测数据集。"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter


SPRITE_NAMES = ("houzi_1.png", "houzi_2.png")


def prepare_sprite(sprite: Image.Image, rng: random.Random) -> Image.Image:
    scale = rng.uniform(0.65, 2.4)
    size = (max(12, int(sprite.width * scale)), max(12, int(sprite.height * scale)))
    sprite = sprite.resize(size, Image.Resampling.NEAREST)
    if rng.random() < 0.5:
        sprite = sprite.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.45:
        sprite = sprite.rotate(
            rng.uniform(-10, 10),
            expand=True,
            resample=Image.Resampling.BICUBIC,
            fillcolor=(0, 0, 0, 0),
        )
    sprite = ImageEnhance.Brightness(sprite).enhance(rng.uniform(0.78, 1.2))
    sprite = ImageEnhance.Contrast(sprite).enhance(rng.uniform(0.85, 1.18))
    if rng.random() < 0.25:
        sprite = sprite.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.7)))
    return sprite


def paste_sprite(canvas: Image.Image, sprite: Image.Image, rng: random.Random):
    if sprite.width >= canvas.width or sprite.height >= canvas.height:
        return None
    x = rng.randint(0, canvas.width - sprite.width)
    y = rng.randint(0, canvas.height - sprite.height)
    canvas.alpha_composite(sprite, (x, y))
    alpha = sprite.getchannel("A").point(lambda value: 255 if value > 24 else 0)
    bbox = alpha.getbbox()
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    return x + left, y + top, x + right, y + bottom


def write_label(path: Path, boxes, size: int) -> None:
    lines = []
    for left, top, right, bottom in boxes:
        cx = ((left + right) / 2) / size
        cy = ((top + bottom) / 2) / size
        width = (right - left) / size
        height = (bottom - top) / size
        lines.append(f"0 {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def make_sample(size: int, background: Image.Image, sprites, distractors, rng: random.Random):
    max_x = max(1, background.width - size)
    max_y = max(1, background.height - size)
    # 建立固定尺寸画布，避免背景尺寸不足或裁剪坐标异常影响标注。
    x = rng.randint(0, max(0, background.width - size))
    y = rng.randint(0, max(0, background.height - size))
    canvas = background.crop((x, y, x + size, y + size)).convert("RGBA")
    if rng.random() < 0.5:
        canvas = ImageEnhance.Brightness(canvas).enhance(rng.uniform(0.82, 1.16))
    if rng.random() < 0.35:
        canvas = ImageEnhance.Contrast(canvas).enhance(rng.uniform(0.86, 1.14))

    # 其他图鉴只作为负样本干扰，不写入标签。
    for _ in range(rng.randint(0, 3)):
        paste_sprite(canvas, prepare_sprite(rng.choice(distractors), rng), rng)

    boxes = []
    target_count = rng.choices((0, 1, 2, 3), weights=(0.08, 0.42, 0.35, 0.15))[0]
    for _ in range(target_count):
        box = paste_sprite(canvas, prepare_sprite(rng.choice(sprites), rng), rng)
        if box:
            boxes.append(box)
    return canvas.convert("RGB"), boxes


def generate_split(root: Path, split: str, count: int, background, sprites, distractors, rng, size: int):
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image, boxes = make_sample(size, background, sprites, distractors, rng)
        stem = f"{split}_{index:05d}"
        image.save(image_dir / f"{stem}.png")
        write_label(label_dir / f"{stem}.txt", boxes, size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("datasets/houzi"))
    parser.add_argument("--background", type=Path, default=Path("validation/face_mobs_real_full_1280.jpg"))
    parser.add_argument("--distractor-dir", type=Path, default=Path("datasets/reference/mob_icons"))
    parser.add_argument(
        "--sprites",
        nargs=2,
        type=Path,
        default=[Path("datasets/houzi/source/houzi_1.png"), Path("datasets/houzi/source/houzi_2.png")],
        help="两张猴子素材，二者会合并为同一个类别",
    )
    parser.add_argument("--train-count", type=int, default=600)
    parser.add_argument("--val-count", type=int, default=80)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()

    source_files = [path for path in args.sprites if path.exists()]
    if len(source_files) != 2:
        raise FileNotFoundError(f"需要两张猴子素材，当前找到 {len(source_files)} 张")
    sprites = [Image.open(path).convert("RGBA") for path in source_files]
    if not args.background.exists():
        raise FileNotFoundError(f"背景图不存在: {args.background}")
    background = Image.open(args.background).convert("RGB")
    distractors = [Image.open(path).convert("RGBA") for path in sorted(args.distractor_dir.glob("*.png"))]
    if not distractors:
        raise FileNotFoundError(f"未找到干扰图: {args.distractor_dir}")

    if args.output.exists():
        for generated_dir in (args.output / "images", args.output / "labels"):
            if generated_dir.exists():
                shutil.rmtree(generated_dir)
    rng = random.Random(args.seed)
    generate_split(args.output, "train", args.train_count, background, sprites, distractors, rng, args.size)
    generate_split(args.output, "val", args.val_count, background, sprites, distractors, rng, args.size)
    (args.output / "data.yaml").write_text(
        f"path: {args.output.resolve()}\ntrain: images/train\nval: images/val\n\nnames:\n  0: 猴子\n",
        encoding="utf-8",
    )
    (args.output / "README.json").write_text(
        json.dumps({
            "classes": {"0": "猴子"},
            "source_variants": 2,
            "horizontal_flip": True,
            "hard_negatives": True,
            "train_images": args.train_count,
            "val_images": args.val_count,
            "seed": args.seed,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"生成完成: {args.output.resolve()} ({args.train_count} train, {args.val_count} val)")


if __name__ == "__main__":
    main()
