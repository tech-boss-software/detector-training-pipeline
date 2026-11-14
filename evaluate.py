"""Evaluate the fine-tuned checkpoint on the held-out test split and render evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import scan
from metrics import evaluate as evaluate_metrics
from train import build_model, configure_runtime, ground_truth, predict


def load_checkpoint(path, device):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model, _ = build_model(len(checkpoint["classes"]), pretrained=False, image_size=checkpoint.get("image_size", 320))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def test_records(data_root, artifacts, checkpoint):
    report, records = scan(Path(data_root))
    if report["fingerprint"] != checkpoint["dataset_fingerprint"]:
        raise ValueError("dataset changed since training; the recorded split no longer applies")
    splits = json.loads((Path(artifacts) / "training_log.json").read_text(encoding="utf-8"))["split"]
    by_id = {r["image_id"]: r for r in records}
    return report, [by_id[i] for i in splits["test"]]


def render(records, predictions, evaluation, classes, output, threshold):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    per_class = evaluation["per_class"]
    fig, axes = plt.subplots(1, len(classes), figsize=(4.2 * len(classes), 4), squeeze=False)
    for axis, name in zip(axes[0], classes):
        curve = per_class[name]["pr_curve_iou50"]
        if curve:
            axis.plot([r for r, _ in curve], [p for _, p in curve], color="#2563eb")
        axis.set_title(f"{name}: AP@0.5 = {per_class[name]['ap50']:.3f}" if per_class[name]["ap50"] is not None else f"{name}: no ground truth")
        axis.set_xlim(0, 1), axis.set_ylim(0, 1.02), axis.set_xlabel("recall"), axis.set_ylabel("precision"), axis.grid(alpha=.3)
    fig.suptitle("Precision-recall on the test split (IoU 0.5), synthetic demonstration data", fontsize=11)
    fig.tight_layout()
    fig.savefig(output / "pr_curves.png", dpi=110)
    plt.close(fig)

    conf = evaluation["confusion"]
    matrix = conf["matrix"]
    fig, axis = plt.subplots(figsize=(5.5, 5))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(conf["labels"]))), axis.set_xticklabels(conf["labels"], rotation=30, ha="right")
    axis.set_yticks(range(len(conf["labels"]))), axis.set_yticklabels(conf["labels"])
    axis.set_xlabel("predicted"), axis.set_ylabel("ground truth")
    peak = max(max(row) for row in matrix) or 1
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            axis.text(j, i, str(value), ha="center", va="center", color="white" if value > peak / 2 else "black")
    axis.set_title(f"Confusion at score >= {conf['score_threshold']}, IoU >= {conf['iou_threshold']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=110)
    plt.close(fig)

    sample = records[:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), squeeze=False)
    fig.suptitle("Test images: ground truth (green dashed) vs predictions at the confusion threshold (red)", fontsize=11)
    for axis, row in zip(axes.flat, sample):
        with Image.open(row["path"]) as image:
            axis.imshow(image.convert("RGB"))
        for obj in row["objects"]:
            x0, y0, x1, y1 = obj["box_xyxy"]
            axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#16a34a", linestyle="--", linewidth=1.5))
        for pred in predictions:
            if pred["image_id"] != row["image_id"] or pred["score"] < threshold:
                continue
            x0, y0, x1, y1 = pred["box_xyxy"]
            axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#dc2626", linewidth=1.5))
            axis.text(x0, max(0, y0 - 3), f"{classes[pred['class_id']]} {pred['score']:.2f}", color="#dc2626", fontsize=8,
                      bbox={"facecolor": "white", "alpha": .7, "pad": 1, "edgecolor": "none"})
        axis.set_title(row["image_id"], fontsize=9)
        axis.axis("off")
    for axis in list(axes.flat)[len(sample):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output / "test_examples.png", dpi=100)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--artifacts", required=True, help="folder containing model.pt and training_log.json")
    parser.add_argument("--threshold", type=float, default=0.5, help="score threshold for the confusion matrix and plots")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args(argv)
    torch = configure_runtime(args.cache_dir, args.threads, 0)
    device = torch.device(args.device)
    artifacts = Path(args.artifacts)
    model, checkpoint = load_checkpoint(artifacts / "model.pt", device)
    report, records = test_records(args.data, artifacts, checkpoint)
    predictions = predict(model, records, device)
    evaluation = evaluate_metrics(predictions, ground_truth(records), checkpoint["classes"], args.threshold)
    evaluation.update({"split": "test", "images": len(records), "ground_truth_count": sum(len(r["objects"]) for r in records),
                       "prediction_count": len(predictions), "model_score_threshold": model.roi_heads.score_thresh,
                       "best_epoch": checkpoint["best_epoch"], "dataset_fingerprint": report["fingerprint"]})
    (artifacts / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    (artifacts / "test_predictions.json").write_text(json.dumps(predictions, indent=1) + "\n", encoding="utf-8")
    render(records, predictions, evaluation, checkpoint["classes"], artifacts, args.threshold)
    print(json.dumps({"map50": evaluation["map50"], "map50_95": evaluation["map50_95"], "images": len(records)}))
    return evaluation


if __name__ == "__main__":
    main()
