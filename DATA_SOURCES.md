# Data and output provenance

## Base checkpoint

Official TorchVision `FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1`, downloaded from the upstream PyTorch URL recorded in `examples/training_log.json` together with the file's SHA-256. Model code and weights are BSD-3-Clause; the checkpoint was trained upstream on COCO, not by Tech Boss. Weights (`.pt`, `.onnx`) are kept outside Git.

## Real dataset: SHWD

Safety Helmet Wearing Dataset by njvisionpower, https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset, MIT licence, 7,581 JPEG images in Pascal VOC layout (`VOC2028`) with classes `hat` (helmet worn) and `person` (bare head); part of the images are crowd-counting scenes from SCUT-HEAD with many small heads. Downloaded on 14 November 2026 and kept outside the repository. `convert_voc.py` mapped `hat` to `helmet` and `person` to `head` and took a random 1,500-image subset (seed 0): 1,655 helmet and 21,966 head boxes, no unmapped names, no dropped boxes (`examples/shwd-model/conversion_report.json`). The split (seed 0) is 1,050 / 225 / 225 images and the label fingerprint is `57e033c05e4f0de2`.

Executed on 14 November 2026 on the same laptop CPU: train 6442 s for 5 epochs at `--image-size 512`, evaluate 75 s, export 14 s; commands in `examples/shwd-model/pipeline.json`. `examples/shwd-model/` holds `training_log.json`, `dataset_report.json`, `evaluation.json`, `evaluation_slices.json` (test images grouped by object count), `export.json`, `pr_curves.png`, `confusion_matrix.png`, `test_examples.jpg`, `inference/` (ONNX-Runtime-only predictions on three test images) and the generated `MODEL_CARD.md`.

## Synthetic dataset

The dataset is generated, not collected: `synthetic.py` draws 1 to 4 flat circles, squares and triangles per 320x320 image over a coloured background with distractor lines, dots, optional blur and seeded pixel noise, and writes YOLO-format labels from the exact geometry it drew. Seed 7, 200 images; the exact per-class counts and the SHA-256 fingerprint of all labels are in `examples/dataset_report.json`. Because the labels come from the generator, there is no annotation error in the ground truth, which is one more reason the scores do not transfer to real photographs. The images are not committed; the generator reproduces them byte-for-byte.

## Executed synthetic run

The committed `examples/` root files were produced by one actual local run of `pipeline.py` on 13 November 2026 on a laptop CPU (Python 3.12.14, torch 2.15.0+cpu, torchvision 0.29.2+cpu, onnx 1.23.2, onnxruntime 1.32.0); exact timings and commands are in `examples/pipeline.json`. The command was:

```powershell
python pipeline.py --output artifacts --count 200 --epochs 8 --threads 4 --cache-dir 'D:\Tech Boss\cv-cache'
```

- `training_log.json`: config, split, per-epoch loss and validation mAP@0.5, best epoch, checkpoint provenance.
- `dataset_report.json`: validation report and fingerprint.
- `evaluation.json`, `test_predictions.json`: held-out test metrics and the raw predictions behind them.
- `pr_curves.png`, `confusion_matrix.png`, `test_examples.png`: rendered with Matplotlib from the JSON and the test images.
- `export.json`: ONNX export details, PyTorch/ONNX Runtime parity and latency.
- `inference/`: ONNX-Runtime-only predictions and annotated PNGs (drawn with Pillow) on three test images.
- `examples/MODEL_CARD-synthetic.md` was generated from these files by `model_card.py`; the root `MODEL_CARD.md` is the SHWD card.

Latency figures are from this one machine and run; they vary with CPU, thread count and background load.
