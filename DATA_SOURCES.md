# Data and output provenance

## Base checkpoint

Official TorchVision `FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1`, downloaded from the upstream PyTorch URL recorded in `examples/training_log.json` together with the file's SHA-256. Model code and weights are BSD-3-Clause; the checkpoint was trained upstream on COCO, not by Tech Boss. Weights (`.pt`, `.onnx`) are kept outside Git.

## Dataset

The dataset is generated, not collected: `synthetic.py` draws 1 to 4 flat circles, squares and triangles per 320x320 image over a coloured background with distractor lines, dots, optional blur and seeded pixel noise, and writes YOLO-format labels from the exact geometry it drew. Seed 7, 200 images; the exact per-class counts and the SHA-256 fingerprint of all labels are in `examples/dataset_report.json`. Because the labels come from the generator, there is no annotation error in the ground truth, which is one more reason the scores do not transfer to real photographs. The images are not committed; the generator reproduces them byte-for-byte.

## Executed run

The committed `examples/` files were produced by one actual local run of `pipeline.py` on 13 November 2026 on a laptop CPU (Python 3.12.14, torch 2.15.0+cpu, torchvision 0.29.2+cpu, onnx 1.23.2, onnxruntime 1.32.0); exact timings and commands are in `examples/pipeline.json`. The command was:

```powershell
python pipeline.py --output artifacts --count 200 --epochs 8 --threads 4 --cache-dir 'D:\Tech Boss\cv-cache'
```

- `training_log.json`: config, split, per-epoch loss and validation mAP@0.5, best epoch, checkpoint provenance.
- `dataset_report.json`: validation report and fingerprint.
- `evaluation.json`, `test_predictions.json`: held-out test metrics and the raw predictions behind them.
- `pr_curves.png`, `confusion_matrix.png`, `test_examples.png`: rendered with Matplotlib from the JSON and the test images.
- `export.json`: ONNX export details, PyTorch/ONNX Runtime parity and latency.
- `inference/`: ONNX-Runtime-only predictions and annotated PNGs (drawn with Pillow) on three test images.
- `MODEL_CARD.md` at the repository root was generated from these files by `model_card.py`.

Latency figures are from this one machine and run; they vary with CPU, thread count and background load.
