"""Deterministic synthetic shape dataset in YOLO layout (images/, labels/, classes.txt).

The scene sampler is pure Python and unit-tested; rendering uses Pillow. The data is
authored, not collected: it proves the pipeline mechanics, not real-world accuracy.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from boxes import iou, to_yolo_line

CLASSES = ["circle", "square", "triangle"]
SIZE = 320


def sample_scene(rng: random.Random, size=SIZE, min_objects=1, max_objects=4, min_side=28, max_side=110, max_overlap=0.15):
    """Return a list of {class_id, box_xyxy, color} that do not overlap by more than max_overlap IoU."""
    objects = []
    wanted = rng.randint(min_objects, max_objects)
    attempts = 0
    while len(objects) < wanted and attempts < 200:
        attempts += 1
        side = rng.randint(min_side, max_side)
        x = rng.randint(0, size - side)
        y = rng.randint(0, size - side)
        box = [float(x), float(y), float(x + side), float(y + side)]
        if any(iou(box, other["box_xyxy"]) > max_overlap for other in objects):
            continue
        color = tuple(rng.randint(30, 225) for _ in range(3))
        objects.append({"class_id": rng.randrange(len(CLASSES)), "box_xyxy": box, "color": color})
    return objects


def render_scene(objects, rng: random.Random, size=SIZE):
    from PIL import Image, ImageDraw, ImageFilter

    base = tuple(rng.randint(120, 245) for _ in range(3))
    image = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(image)
    # Distractors: lines and tiny dots that are not labelled objects.
    for _ in range(rng.randint(2, 6)):
        p = [rng.randint(0, size) for _ in range(4)]
        draw.line(p, fill=tuple(max(0, c - rng.randint(20, 60)) for c in base), width=rng.randint(1, 3))
    for _ in range(rng.randint(5, 20)):
        cx, cy = rng.randint(0, size), rng.randint(0, size)
        draw.ellipse([cx, cy, cx + 3, cy + 3], fill=tuple(rng.randint(0, 255) for _ in range(3)))
    for item in objects:
        x0, y0, x1, y1 = item["box_xyxy"]
        name = CLASSES[item["class_id"]]
        if name == "circle":
            draw.ellipse([x0, y0, x1, y1], fill=item["color"])
        elif name == "square":
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=item["color"])
        else:
            draw.polygon([(x0, y1 - 1), (x1 - 1, y1 - 1), ((x0 + x1) / 2, y0)], fill=item["color"])
    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    noise_seed = rng.randint(0, 2**31 - 1)
    return _add_noise(image, noise_seed, amplitude=rng.randint(4, 18))


def _add_noise(image, seed, amplitude):
    import numpy as np
    from PIL import Image

    # A seeded NumPy generator keeps the dataset byte-identical across runs.
    noise = np.random.default_rng(seed).integers(-amplitude, amplitude + 1, size=(image.height, image.width, 3))
    pixels = np.asarray(image, dtype=np.int16) + noise
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")


def generate(root: Path, count=200, seed=7, size=SIZE):
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    rng = random.Random(seed)
    summary = {"seed": seed, "count": count, "size": size, "classes": CLASSES, "objects_per_class": [0] * len(CLASSES)}
    for index in range(count):
        objects = sample_scene(rng, size)
        image = render_scene(objects, rng, size)
        stem = f"synthetic_{index:04d}"
        image.save(root / "images" / f"{stem}.png")
        lines = [to_yolo_line(o["class_id"], o["box_xyxy"], size, size) for o in objects]
        (root / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for item in objects:
            summary["objects_per_class"][item["class_id"]] += 1
    (root / "synthetic.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    summary = generate(Path(args.output), args.count, args.seed)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
