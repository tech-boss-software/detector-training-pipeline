"""Fine-tune the official TorchVision Faster R-CNN MobileNetV3-Large 320 FPN on a YOLO-layout dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

from datasets import require_clean, scan, split
from metrics import evaluate as evaluate_metrics

MODEL_NAME = "fasterrcnn_mobilenet_v3_large_320_fpn"
WEIGHTS_NAME = "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1"


def configure_runtime(cache_dir, threads, seed):
    if cache_dir:
        os.environ["TORCH_HOME"] = str(Path(cache_dir).resolve())
    import torch

    torch.set_num_threads(int(threads))
    torch.manual_seed(seed)
    random.seed(seed)
    return torch


def load_image_tensor(path):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def build_model(class_count, pretrained=True, image_size=320):
    import torchvision
    from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1 if pretrained else None
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights, weights_backbone=None if not pretrained else "DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Index 0 stays background, so the head has class_count + 1 outputs.
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, class_count + 1)
    # The internal resize decides how many pixels a small object keeps. The checkpoint
    # was trained at 320; a larger size costs latency but makes helmets, labels and
    # defects detectable. The value is stored in model.pt and baked into the ONNX graph.
    model.transform.min_size = (int(image_size),)
    model.transform.max_size = int(image_size) * 2
    return model, weights


def checkpoint_provenance(weights):
    if weights is None:
        return {"name": None, "url": None, "file": None, "sha256": None}
    import torch

    filename = Path(weights.url).name
    path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {"name": WEIGHTS_NAME, "url": weights.url, "file": filename, "sha256": digest.hexdigest()}


class DetectionDataset:
    def __init__(self, records, train, seed):
        self.records = records
        self.train = train
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch

        row = self.records[index]
        image = load_image_tensor(row["path"])
        boxes = torch.tensor([o["box_xyxy"] for o in row["objects"]], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([o["class_id"] + 1 for o in row["objects"]], dtype=torch.int64)
        if self.train and self.rng.random() < 0.5:
            image = image.flip(-1)
            width = float(row["width"])
            boxes = torch.stack([width - boxes[:, 2], boxes[:, 1], width - boxes[:, 0], boxes[:, 3]], dim=1) if len(boxes) else boxes
        return image, {"boxes": boxes, "labels": labels}


def collate(batch):
    return [item[0] for item in batch], [item[1] for item in batch]


def predict(model, records, device, batch_size=4):
    """Run inference and return flat prediction records with class_id = label - 1."""
    import torch

    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            rows = records[start:start + batch_size]
            images = [load_image_tensor(r["path"]).to(device) for r in rows]
            outputs = model(images)
            for row, output in zip(rows, outputs):
                for box, label, score in zip(output["boxes"].tolist(), output["labels"].tolist(), output["scores"].tolist()):
                    if label == 0 or box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    predictions.append({"image_id": row["image_id"], "class_id": int(label) - 1,
                                        "box_xyxy": [float(v) for v in box], "score": float(score)})
    return predictions


def ground_truth(records):
    return [{"image_id": r["image_id"], "class_id": o["class_id"], "box_xyxy": o["box_xyxy"]}
            for r in records for o in r["objects"]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="dataset root with images/, labels/, classes.txt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-dir", default=None, help="Torch hub cache outside the repository")
    parser.add_argument("--no-pretrained", action="store_true", help="train from random initialisation (slow, for comparison)")
    parser.add_argument("--image-size", type=int, default=320, help="shorter side after the model's internal resize; raise it for small objects")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N images (smoke runs)")
    parser.add_argument("--origin", default="see DATA_SOURCES.md", help="one-line provenance note for the dataset")
    args = parser.parse_args(argv)

    torch = configure_runtime(args.cache_dir, args.threads, args.seed)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    report, records = scan(Path(args.data))
    require_clean(report)
    if args.limit:
        records = records[:args.limit]
    report["origin"] = args.origin
    splits = split(records, args.seed)
    by_id = {r["image_id"]: r for r in records}
    train_records = [by_id[i] for i in splits["train"]]
    val_records = [by_id[i] for i in splits["val"]]

    model, weights = build_model(len(report["classes"]), pretrained=not args.no_pretrained, image_size=args.image_size)
    model.to(device)
    provenance = checkpoint_provenance(weights)
    loader = torch.utils.data.DataLoader(DetectionDataset(train_records, True, args.seed), batch_size=args.batch_size,
                                         shuffle=True, collate_fn=collate, num_workers=0,
                                         generator=torch.Generator().manual_seed(args.seed))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[max(1, int(args.epochs * 0.7))], gamma=0.1)

    log = []
    best_state, best_map, best_epoch = None, -1.0, None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        epoch_started = time.perf_counter()
        for images, targets in loader:
            images = [i.to(device) for i in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            optimizer.step()
            total += loss.item()
            count += 1
        scheduler.step()
        val_metrics = evaluate_metrics(predict(model, val_records, device), ground_truth(val_records), report["classes"])
        val_map = val_metrics["map50"]
        log.append({"epoch": epoch, "train_loss": total / max(count, 1), "val_map50": val_map,
                    "lr": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - epoch_started})
        print(json.dumps(log[-1]))
        if val_map is not None and val_map > best_map:
            best_map, best_epoch = val_map, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:  # validation had no ground truth for any class
        best_state, best_epoch, best_map = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, args.epochs, None

    checkpoint = {"state_dict": best_state, "classes": report["classes"], "model": MODEL_NAME,
                  "weights": provenance["name"], "best_epoch": best_epoch, "dataset_fingerprint": report["fingerprint"],
                  "image_size": args.image_size}
    torch.save(checkpoint, output / "model.pt")
    import torchvision

    training_log = {
        "model": {"name": MODEL_NAME, "weights": provenance["name"], "checkpoint": provenance, "classes": report["classes"]},
        "config": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "momentum": 0.9,
                   "weight_decay": 5e-4, "seed": args.seed, "augmentation": "random horizontal flip (p=0.5)",
                   "pretrained": not args.no_pretrained, "limit": args.limit, "image_size": args.image_size},
        "environment": {"device": str(device), "threads": args.threads, "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "python": platform.python_version(), "platform": platform.platform()},
        "split": splits, "epochs": log, "best_epoch": best_epoch, "best_val_map50": best_map,
        "seconds": time.perf_counter() - started,
    }
    (output / "training_log.json").write_text(json.dumps(training_log, indent=2) + "\n", encoding="utf-8")
    (output / "dataset_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"best epoch {best_epoch} val mAP@0.5 {best_map}; wrote {output / 'model.pt'}")
    return training_log


if __name__ == "__main__":
    main()
