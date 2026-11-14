"""Export the fine-tuned detector to ONNX, verify parity against PyTorch and measure CPU latency."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

from datasets import scan
from evaluate import load_checkpoint, test_records
from train import configure_runtime, load_image_tensor

OPSET = 11


def _onnx_safe_decode(box_coder):
    """Equivalent of TorchVision BoxCoder.decode whose reshapes survive an empty proposal set.

    TorchVision writes ``rel_codes.reshape(box_sum, -1)``; when the RPN finds no
    proposals (blank or noise input) that becomes an ONNX Reshape to ``{0, -1}``,
    which ONNX Runtime rejects. Putting ``-1`` on the row axis keeps the column count
    explicit, so the empty case reshapes to ``{0, C}`` and the graph returns no boxes.
    """
    import torch

    def decode(rel_codes, boxes):
        concat_boxes = torch.cat(boxes, dim=0)
        columns = rel_codes.shape[1]
        pred_boxes = box_coder.decode_single(rel_codes.reshape(-1, columns), concat_boxes)
        return pred_boxes.reshape(-1, columns // 4, 4)

    return decode


def trace_image(records):
    """Pick a labelled image with objects: tracing on noise follows the empty-proposal branch."""
    for row in records:
        if row["objects"]:
            return load_image_tensor(row["path"])
    raise ValueError("no training image with objects to trace the export on")


def export_onnx(model, path, classes, sample):
    import copy

    import onnx
    import torch

    export_model = copy.deepcopy(model).eval()
    export_model.roi_heads.box_coder.decode = _onnx_safe_decode(export_model.roi_heads.box_coder)
    # The TorchScript-based exporter handles TorchVision's NMS and RoI ops; the dynamo
    # exporter stops on a data-dependent guard inside batched NMS, so it is not used.
    torch.onnx.export(export_model, ([sample],), str(path), opset_version=OPSET, dynamo=False,
                      input_names=["image"], output_names=["boxes", "labels", "scores"],
                      dynamic_axes={"image": {1: "height", 2: "width"}, "boxes": {0: "detections"},
                                    "labels": {0: "detections"}, "scores": {0: "detections"}})
    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)
    # Self-describing model: class names and preprocessing travel with the file.
    for key, value in {"classes": json.dumps(classes), "input": "RGB float32 CHW in [0,1], any size; labels are 1-based, 0 = background",
                       "producer": "Tech Boss detector-training-pipeline"}.items():
        entry = onnx_model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(onnx_model, str(path))
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {"exporter": "torch.onnx.export (TorchScript-based, dynamo=False), traced on a labelled training image",
            "opset": OPSET, "decode_patch": "BoxCoder.decode reshapes use -1 on the row axis so empty proposal sets export",
            "size_mb": Path(path).stat().st_size / 1e6, "sha256": digest, "onnx": onnx.__version__,
            "inputs": ["image"], "outputs": ["boxes", "labels", "scores"]}


def session_for(path, threads):
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = int(threads)
    return onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def parity(model, session, records, limit=8):
    import numpy as np
    import torch

    worst_box, worst_score, mismatches, count_diffs = 0.0, 0.0, 0, 0
    used = 0
    with torch.no_grad():
        for row in records[:limit]:
            tensor = load_image_tensor(row["path"])
            eager = model([tensor])[0]
            boxes, labels, scores = session.run(None, {"image": tensor.numpy()})
            n = min(len(eager["scores"]), len(scores))
            count_diffs += abs(len(eager["scores"]) - len(scores))
            if n:
                worst_box = max(worst_box, float(np.abs(eager["boxes"][:n].numpy() - boxes[:n]).max()))
                worst_score = max(worst_score, float(np.abs(eager["scores"][:n].numpy() - scores[:n]).max()))
                mismatches += int((eager["labels"][:n].numpy() != labels[:n]).sum())
            used += 1
    # A blank image must yield zero detections, not a runtime error, in both runtimes.
    blank = torch.zeros(3, 320, 320)
    with torch.no_grad():
        eager_blank = int(len(model([blank])[0]["scores"]))
    onnx_blank = int(len(session.run(None, {"image": blank.numpy()})[2]))
    return {"images": used, "max_box_abs_diff": worst_box, "max_score_abs_diff": worst_score,
            "label_mismatches": mismatches, "detection_count_differences": count_diffs,
            "blank_image_detections": {"torch": eager_blank, "onnxruntime": onnx_blank},
            "note": "outputs are compared in the model's own confidence order on the same input tensor"}


def latency(model, session, record, threads, runs=20, warmup=3):
    import torch

    tensor = load_image_tensor(record["path"])
    array = tensor.numpy()

    def timed(function):
        for _ in range(warmup):
            function()
        samples = []
        for _ in range(runs):
            started = time.perf_counter()
            function()
            samples.append((time.perf_counter() - started) * 1000)
        samples.sort()
        return statistics.median(samples), samples[int(0.95 * (len(samples) - 1))]

    with torch.no_grad():
        torch_median, torch_p95 = timed(lambda: model([tensor]))
    ort_median, ort_p95 = timed(lambda: session.run(None, {"image": array}))
    return {"width": int(tensor.shape[2]), "height": int(tensor.shape[1]), "threads": threads, "runs": runs, "warmup": warmup,
            "torch_median_ms": torch_median, "torch_p95_ms": torch_p95,
            "onnxruntime_median_ms": ort_median, "onnxruntime_p95_ms": ort_p95,
            "cpu": platform.processor() or platform.machine(), "image": record["image_id"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args(argv)
    torch = configure_runtime(args.cache_dir, args.threads, 0)
    import onnxruntime

    artifacts = Path(args.artifacts)
    model, checkpoint = load_checkpoint(artifacts / "model.pt", torch.device("cpu"))
    _, records = test_records(args.data, artifacts, checkpoint)
    splits = json.loads((artifacts / "training_log.json").read_text(encoding="utf-8"))["split"]
    all_records = {r["image_id"]: r for r in scan(Path(args.data))[1]}
    sample = trace_image([all_records[i] for i in splits["train"]])
    onnx_path = artifacts / "model.onnx"
    info = export_onnx(model, onnx_path, checkpoint["classes"], sample)
    session = session_for(onnx_path, args.threads)
    info["onnxruntime"] = onnxruntime.__version__
    info["parity"] = parity(model, session, records)
    info["latency"] = latency(model, session, records[0], args.threads, args.runs)
    info["classes"] = checkpoint["classes"]
    (artifacts / "export.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: info[k] for k in ("size_mb", "parity", "latency")}))
    return info


if __name__ == "__main__":
    main()
