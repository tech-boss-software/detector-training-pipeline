"""YOLO-layout dataset validation and deterministic splitting (no framework imports)."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from boxes import parse_yolo_line

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def read_classes(root: Path) -> list[str]:
    path = Path(root) / "classes.txt"
    if not path.is_file():
        raise ValueError(f"missing classes.txt in {root}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names or len(set(names)) != len(names):
        raise ValueError("classes.txt must list at least one unique class name per line")
    return names


def image_size(path: Path):
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def scan(root: Path, size_reader=image_size):
    """Validate every image/label pair and return a report plus per-image records.

    Problems are collected, not raised one at a time, so a client dataset can be
    fixed in one pass. Missing label files mean 'no objects' (a negative image),
    which is valid and reported separately.
    """
    root = Path(root)
    classes = read_classes(root)
    images_dir, labels_dir = root / "images", root / "labels"
    if not images_dir.is_dir():
        raise ValueError(f"missing images/ directory in {root}")
    records, problems, negatives = [], [], 0
    per_class = [0] * len(classes)
    for image_path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        try:
            width, height = size_reader(image_path)
        except Exception as error:  # unreadable image is a dataset problem, not a crash
            problems.append({"image": image_path.name, "problem": f"unreadable image: {error}"})
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        objects = []
        if label_path.is_file():
            for number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    class_id, box = parse_yolo_line(line, width, height, len(classes))
                except ValueError as error:
                    problems.append({"image": image_path.name, "line": number, "problem": str(error)})
                    continue
                objects.append({"class_id": class_id, "box_xyxy": box})
                per_class[class_id] += 1
        else:
            negatives += 1
        records.append({"image_id": image_path.name, "path": str(image_path), "width": width, "height": height, "objects": objects})
    orphan_labels = sorted(p.name for p in labels_dir.glob("*.txt")) if labels_dir.is_dir() else []
    stems = {r["image_id"].rsplit(".", 1)[0] for r in records}
    orphan_labels = [name for name in orphan_labels if name[:-4] not in stems]
    report = {
        "root": str(root), "classes": classes, "images": len(records), "negative_images": negatives,
        "objects": sum(per_class), "objects_per_class": dict(zip(classes, per_class)),
        "problems": problems, "orphan_labels": orphan_labels,
        "fingerprint": _fingerprint(records),
    }
    return report, records


def _fingerprint(records):
    digest = hashlib.sha256()
    for row in records:
        digest.update(json.dumps({k: row[k] for k in ("image_id", "width", "height", "objects")}, sort_keys=True).encode())
    return digest.hexdigest()


def split(records, seed=0, fractions=(0.7, 0.15, 0.15)):
    """Deterministic shuffle split into train/val/test by image, never by object."""
    if abs(sum(fractions) - 1.0) > 1e-9 or any(f < 0 for f in fractions):
        raise ValueError("fractions must be nonnegative and sum to 1")
    ids = sorted(r["image_id"] for r in records)
    if len(ids) < 3:
        raise ValueError("need at least three images to split")
    random.Random(seed).shuffle(ids)
    n_val = max(1, round(len(ids) * fractions[1]))
    n_test = max(1, round(len(ids) * fractions[2]))
    n_train = len(ids) - n_val - n_test
    if n_train < 1:
        raise ValueError("each split must receive at least one image")
    train, val, test = ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]
    return {"seed": seed, "fractions": list(fractions), "train": train, "val": val, "test": test}


def require_clean(report):
    if report["problems"]:
        preview = "; ".join(f"{p['image']}: {p['problem']}" for p in report["problems"][:5])
        raise ValueError(f"{len(report['problems'])} label problems, fix them before training: {preview}")
    if report["images"] == 0:
        raise ValueError("dataset has no images")
    if report["objects"] == 0:
        raise ValueError("dataset has no labelled objects")
