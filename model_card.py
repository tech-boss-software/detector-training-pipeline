"""Render MODEL_CARD.md from the JSON produced by train, evaluate and export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(value, digits=3):
    if value is None:
        return "undefined"
    return f"{value:.{digits}f}"


def render(training, evaluation, export, dataset_report):
    classes = dataset_report["classes"]
    lines = ["# Model card: fine-tuned detector", ""]
    lines += ["**Project owner:** Tech Boss. **Base checkpoint:** official TorchVision "
              f"`{training['model']['name']}` ({training['model']['weights']}). "
              "**Fine-tuning performed by Tech Boss:** yes, on the dataset described below.", ""]
    lines += ["## Intended use", "",
              "Detect the classes listed under Data on images like the ones it was trained on, on a CPU machine, "
              "behind a human-review threshold. The numbers below describe this dataset only; they say nothing "
              "about another camera, site or labelling policy until the pipeline is re-run on that data.", ""]
    lines += ["## Data", "",
              f"- Source: `{dataset_report['root']}` ({dataset_report.get('origin', 'see DATA_SOURCES.md')})",
              f"- Images: {dataset_report['images']} ({dataset_report['negative_images']} without objects), "
              f"objects: {dataset_report['objects']}",
              "- Objects per class: " + ", ".join(f"{k} {v}" for k, v in dataset_report["objects_per_class"].items()),
              f"- Split (seed {training['split']['seed']}): train {len(training['split']['train'])}, "
              f"val {len(training['split']['val'])}, test {len(training['split']['test'])} images; the test split "
              "was not used for model selection",
              f"- Dataset fingerprint: `{dataset_report['fingerprint'][:16]}`", ""]
    cfg = training["config"]
    lines += ["## Training", "",
              f"- Epochs {cfg['epochs']}, batch {cfg['batch_size']}, SGD lr {cfg['lr']}, momentum {cfg['momentum']}, "
              f"weight decay {cfg['weight_decay']}, seed {cfg['seed']}, device {training['environment']['device']}, "
              f"threads {training['environment']['threads']}",
              f"- Augmentation: {cfg['augmentation']}",
              f"- Internal resize: shorter side {cfg.get('image_size', 320)} px (the base checkpoint was trained at 320)",
              f"- Best epoch by validation mAP@0.5: {training['best_epoch']} "
              f"(val mAP@0.5 {_fmt(training['best_val_map50'])})",
              f"- Wall time: {training['seconds']:.0f} s; software: torch {training['environment']['torch']}, "
              f"torchvision {training['environment']['torchvision']}", ""]
    lines += ["| Epoch | Train loss | Val mAP@0.5 |", "|---|---|---|"]
    for row in training["epochs"]:
        lines.append(f"| {row['epoch']} | {row['train_loss']:.4f} | {_fmt(row['val_map50'])} |")
    lines.append("")
    lines += ["## Evaluation on the held-out test split", "",
              f"- mAP@0.5: **{_fmt(evaluation['map50'])}**, mAP@[0.5:0.95]: **{_fmt(evaluation['map50_95'])}** "
              f"over {evaluation['classes_with_ground_truth']} classes with ground truth",
              f"- Method: {evaluation['method']}", "",
              "| Class | Ground truth | Predictions | AP@0.5 | AP@[0.5:0.95] |", "|---|---|---|---|---|"]
    for name, row in evaluation["per_class"].items():
        lines.append(f"| {name} | {row['ground_truth']} | {row['predictions']} | {_fmt(row['ap50'])} | {_fmt(row['ap50_95'])} |")
    conf = evaluation["confusion"]
    lines += ["", f"Confusion matrix at score >= {conf['score_threshold']} and IoU >= {conf['iou_threshold']} "
              "(rows = ground truth, columns = predictions):", "",
              "| | " + " | ".join(conf["labels"]) + " |", "|---|" + "---|" * len(conf["labels"])]
    for label, row in zip(conf["labels"], conf["matrix"]):
        lines.append(f"| **{label}** | " + " | ".join(str(v) for v in row) + " |")
    lines.append("")
    if export:
        parity = export["parity"]
        lat = export["latency"]
        lines += ["## ONNX export", "",
                  f"- Exporter: {export['exporter']}, opset {export['opset']}, dynamic image size; "
                  f"file size {export['size_mb']:.1f} MB, SHA-256 `{export['sha256'][:16]}` (weights are not committed)",
                  f"- Parity on {parity['images']} test images: max box difference {parity['max_box_abs_diff']:.5f} px, "
                  f"max score difference {parity['max_score_abs_diff']:.6f}, label mismatches {parity['label_mismatches']}",
                  f"- Latency (single image, {lat['width']}x{lat['height']}, {lat['threads']} CPU threads, "
                  f"{lat['runs']} runs after warm-up): PyTorch eager median {lat['torch_median_ms']:.1f} ms "
                  f"(p95 {lat['torch_p95_ms']:.1f}), ONNX Runtime median {lat['onnxruntime_median_ms']:.1f} ms "
                  f"(p95 {lat['onnxruntime_p95_ms']:.1f}) on {lat['cpu']}",
                  "- Latency is a local measurement on one machine; it is not a benchmark of hardware.", ""]
    lines += ["## Licence note", "",
              "TorchVision model code and the COCO-pretrained checkpoint are BSD-3-Clause licensed; original "
              "pipeline code is MIT. No AGPL component is used, so the exported model can be deployed in a "
              "client product without copyleft obligations. Confirm the client dataset's own terms separately.", ""]
    origin = str(dataset_report.get("origin", ""))
    synthetic = "synthetic" in origin.lower()
    data_note = ("- The dataset is synthetic: flat shapes on noisy backgrounds. High scores here do not transfer to "
                 "photographs, occlusion, small objects, motion blur or class imbalance." if synthetic else
                 "- Scores describe this dataset's images and labelling; crowded scenes, tiny objects and a different "
                 "label policy move them, so inspect per-class and per-slice results before trusting the mean.")
    lines += ["## Limitations", "", data_note,
              "- Scores are not calibrated probabilities; the confusion matrix threshold is a choice, not a fact.",
              "- Greedy matching without crowd/ignore semantics differs slightly from pycocotools on crowded scenes.",
              f"- CPU training resizes the shorter side to {cfg.get('image_size', 320)} px; objects of a few pixels are lost.",
              "- No tracking, segmentation, drift monitoring or serving layer is included.", ""]
    lines += ["## Before client deployment", "",
              "Re-run the pipeline on representative licensed data with every object labelled, inspect per-class and "
              "per-slice results, compare the ONNX latency on the target hardware, choose thresholds with a human-review "
              "boundary, and keep the dataset fingerprint and metrics JSON with the delivered model.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, help="folder with training_log.json, evaluation.json, export.json, dataset_report.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    folder = Path(args.artifacts)
    load = lambda name: json.loads((folder / name).read_text(encoding="utf-8"))
    export = load("export.json") if (folder / "export.json").is_file() else None
    text = render(load("training_log.json"), load("evaluation.json"), export, load("dataset_report.json"))
    Path(args.output).write_text(text, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
