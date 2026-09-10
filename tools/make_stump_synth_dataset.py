from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


TARGET_SPRITES = {
    0: [
        "datasets/mob_stumps/images/train/mob_1130100_斧木妖_orig.png",
        "datasets/mob_stumps/images/train/mob_1130100_斧木妖_flip.png",
    ],
    1: [
        "datasets/mob_stumps/images/train/mob_130100_木妖_orig.png",
        "datasets/mob_stumps/images/train/mob_130100_木妖_flip.png",
    ],
}


def load_rgba(path: str) -> Image.Image:
    return Image.open(path).convert("RGBA")


def random_background_crop(background: Image.Image, size: int, rng: random.Random) -> Image.Image:
    max_x = max(1, background.width - size)
    max_y = max(1, background.height - size)
    x = rng.randrange(max_x + 1)
    y = rng.randrange(max_y + 1)
    return background.crop((x, y, x + size, y + size))


def adjust_background(canvas: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.65:
        canvas = ImageEnhance.Brightness(canvas).enhance(rng.uniform(0.80, 1.18))
    if rng.random() < 0.55:
        canvas = ImageEnhance.Color(canvas).enhance(rng.uniform(0.82, 1.15))
    if rng.random() < 0.35:
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0, 0.6)))
    return canvas


def paste_sprite(canvas: Image.Image, sprite: Image.Image, size: int) -> tuple[int, int, int, int]:
    scale = size / max(sprite.width, sprite.height)
    scaled = sprite.resize((max(1, int(sprite.width * scale)), max(1, int(sprite.height * scale))))
    x = rng.randrange(0, max(1, canvas.width - scaled.width + 1))
    y = rng.randrange(0, max(1, canvas.height - scaled.height + 1))
    canvas.alpha_composite(scaled, (x, y))
    return x, y, scaled.width, scaled.height


def write_label(path: Path, boxes: list[tuple[int, int, int, int, int]], width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for class_id, x, y, w, h in boxes:
        cx = (x + w / 2) / width
        cy = (y + h / 2) / height
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def generate_split(
    root: Path,
    split: str,
    count: int,
    background: Image.Image,
    targets: dict[int, list[Image.Image]],
    distractors: list[Image.Image],
    rng: random.Random,
) -> None:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for index in range(count):
        canvas = random_background_crop(background, args.size, rng).convert("RGBA")
        canvas = adjust_background(canvas, rng)
        boxes = []
        positive_count = 1 if split == "val" else rng.randint(1, 3)
        classes = list(targets)
        rng.shuffle(classes)
        for class_id in classes[:positive_count]:
            sprite = rng.choice(targets[class_id])
            box = paste_sprite(canvas, sprite, rng.randint(48, 104))
            boxes.append((class_id, *box))

        for _ in range(rng.randint(0, 2) if split == "train" else 1):
            paste_sprite(canvas, rng.choice(distractors), rng.randint(32, 72))

        visible = Image.new("RGB", canvas.size, (0, 0, 0))
        visible.paste(canvas, (0, 0), canvas)
        image_name = f"{split}_{index:05d}.png"
        visible.save(image_dir / image_name)
        write_label(label_dir / f"{split}_{index:05d}.txt", boxes, canvas.width, canvas.height)


def main() -> None:
    global rng, args
    parser = argparse.ArgumentParser(description="生成斧木妖/木妖合成检测数据集")
    parser.add_argument("--dataset", default="datasets/mob_stumps_synth")
    parser.add_argument("--background", default="validation/face_mobs_real_full_1280.jpg")
    parser.add_argument("--distractor-dir", default="datasets/reference/mob_icons")
    parser.add_argument("--train-count", type=int, default=600)
    parser.add_argument("--val-count", type=int, default=48)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    root = Path(args.dataset)
    if root.exists():
        shutil.rmtree(root)

    background = Image.open(args.background).convert("RGB")
    targets = {
        class_id: [load_rgba(path) for path in paths]
        for class_id, paths in TARGET_SPRITES.items()
    }
    excluded = {Path(path).name for paths in TARGET_SPRITES.values() for path in paths}
    distractor_paths = sorted(
        path for path in Path(args.distractor_dir).glob("*.png") if path.name not in excluded
    )
    distractors = [load_rgba(str(path)) for path in distractor_paths]

    generate_split(root, "train", args.train_count, background, targets, distractors, rng)
    generate_split(root, "val", args.val_count, background, targets, distractors, rng)

    names = "{\n  0: 斧木妖\n  1: 木妖\n}"
    (root / "data.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: 斧木妖\n  1: 木妖\n",
        encoding="utf-8",
    )
    print(f"生成完成: {root.resolve()} ({args.train_count} train, {args.val_count} val)")


if __name__ == "__main__":
    main()
