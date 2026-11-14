"""Convert a Pascal VOC dataset (JPEGImages + Annotations XML) to the YOLO layout this pipeline trains on.

    python convert_voc.py --voc path/to/VOC2028 --output artifacts/dataset --classes hat=helmet person=head [--limit N]

Class mapping renames VOC names to the names the downstream project expects (for the Safety
Helmet Wearing Dataset: `hat` is a head with a helmet, `person` is a bare head). Boxes are
clipped to the image, degenerate boxes are dropped and counted, and a report is written.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def convert(voc_root, output, mapping, limit=0, seed=0, copy=True):
    voc_root, output = Path(voc_root), Path(output)
    images_dir = voc_root / "JPEGImages"
    annotations_dir = voc_root / "Annotations"
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        raise ValueError(f"{voc_root} needs JPEGImages/ and Annotations/")
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(mapping.values()))
    (output / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    xml_files = sorted(annotations_dir.glob("*.xml"))
    if limit:
        random.Random(seed).shuffle(xml_files)
        xml_files = sorted(xml_files[:limit])
    report = {"source": str(voc_root), "images": 0, "skipped_images": 0, "objects": {n: 0 for n in names}, "unmapped": {}, "dropped_boxes": 0}
    for xml_path in xml_files:
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename") or (xml_path.stem + ".jpg")
        image_path = images_dir / filename
        if not image_path.exists():
            image_path = next((p for p in images_dir.glob(xml_path.stem + ".*")), None)
        size = root.find("size")
        if image_path is None or size is None:
            report["skipped_images"] += 1
            continue
        width, height = float(size.findtext("width")), float(size.findtext("height"))
        if width <= 0 or height <= 0:
            report["skipped_images"] += 1
            continue
        lines = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name not in mapping:
                report["unmapped"][name] = report["unmapped"].get(name, 0) + 1
                continue
            box = obj.find("bndbox")
            try:
                x0, y0, x1, y1 = (float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            except (TypeError, ValueError, AttributeError):
                report["dropped_boxes"] += 1
                continue
            x0, x1 = max(0.0, min(width, x0)), max(0.0, min(width, x1))
            y0, y1 = max(0.0, min(height, y0)), max(0.0, min(height, y1))
            if x1 - x0 < 1 or y1 - y0 < 1:
                report["dropped_boxes"] += 1
                continue
            class_id = names.index(mapping[name])
            report["objects"][mapping[name]] += 1
            lines.append(f"{class_id} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} {(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}")
        target = output / "images" / image_path.name
        if copy and not target.exists():
            shutil.copyfile(image_path, target)
        (output / "labels" / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        report["images"] += 1
    (output / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voc", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classes", nargs="+", required=True, help="VOC name=new name pairs, in the class order you want")
    parser.add_argument("--limit", type=int, default=0, help="random subset of images (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    mapping = dict(pair.split("=", 1) for pair in args.classes)
    print(json.dumps(convert(args.voc, args.output, mapping, args.limit, args.seed)))


if __name__ == "__main__":
    main()
