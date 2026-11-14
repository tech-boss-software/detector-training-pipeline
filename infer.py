"""Run the exported ONNX detector with ONNX Runtime only (no PyTorch needed) and draw results."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from boxes import clip_box

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(inputs):
    images = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            images.extend(p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        else:
            raise ValueError(f"not a supported image or directory: {path}")
    if not images:
        raise ValueError("no supported images found")
    return images


def load_session(model_path, threads):
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = int(threads)
    session = onnxruntime.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    if "classes" not in meta:
        raise ValueError("model has no 'classes' metadata; export it with export.py")
    return session, json.loads(meta["classes"])


def detect(session, classes, path, threshold):
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
    width, height = image.size
    tensor = np.ascontiguousarray(array.transpose(2, 0, 1))
    started = time.perf_counter()
    boxes, labels, scores = session.run(None, {"image": tensor})
    elapsed = (time.perf_counter() - started) * 1000
    detections = []
    for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        if score < threshold or label == 0:
            continue
        clipped = clip_box(box, width, height)
        if clipped is None:
            continue
        detections.append({"box_xyxy": clipped, "score": float(score), "class_id": int(label) - 1, "label": classes[int(label) - 1]})
    detections.sort(key=lambda d: -d["score"])
    return {"image_id": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "width": width, "height": height,
            "inference_ms": elapsed, "detections": detections}, image


def draw(image, result, output_path):
    from PIL import ImageDraw

    canvas = image.copy()
    drawer = ImageDraw.Draw(canvas)
    for det in result["detections"]:
        x0, y0, x1, y1 = det["box_xyxy"]
        drawer.rectangle([x0, y0, x1, y1], outline="#dc2626", width=2)
        text = f"{det['label']} {det['score']:.2f}"
        tx, ty = x0 + 2, max(0, y0 - 12)
        tw = drawer.textlength(text) + 4
        drawer.rectangle([tx - 2, ty, tx + tw, ty + 12], fill="#dc2626")
        drawer.text((tx, ty), text, fill="white")
    if not result["detections"]:
        drawer.text((4, 4), "no detections at this threshold", fill="#dc2626")
    canvas.save(output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model.onnx written by export.py")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    session, classes = load_session(args.model, args.threads)
    results = []
    for path in collect_images(args.input):
        result, image = detect(session, classes, path, args.threshold)
        draw(image, result, output / f"{path.stem}_detections.png")
        results.append(result)
    summary = {"model": str(args.model), "model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
               "classes": classes, "threshold": args.threshold, "threads": args.threads, "images": results}
    (output / "predictions.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{"image": r["image_id"], "detections": len(r["detections"]), "ms": round(r["inference_ms"], 1)} for r in results]))
    return summary


if __name__ == "__main__":
    main()
