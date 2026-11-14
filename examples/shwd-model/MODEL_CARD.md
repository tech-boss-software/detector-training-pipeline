# Model card: fine-tuned detector

**Project owner:** Tech Boss. **Base checkpoint:** official TorchVision `fasterrcnn_mobilenet_v3_large_320_fpn` (FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1). **Fine-tuning performed by Tech Boss:** yes, on the dataset described below.

## Intended use

Detect the classes listed under Data on images like the ones it was trained on, on a CPU machine, behind a human-review threshold. The numbers below describe this dataset only; they say nothing about another camera, site or labelling policy until the pipeline is re-run on that data.

## Data

- Source: `artifacts\shwd-dataset` (Safety Helmet Wearing Dataset (SHWD, MIT licence), random 1500-image subset converted from Pascal VOC)
- Images: 1500 (0 without objects), objects: 23621
- Objects per class: helmet 1655, head 21966
- Split (seed 0): train 1050, val 225, test 225 images; the test split was not used for model selection
- Dataset fingerprint: `57e033c05e4f0de2`

## Training

- Epochs 5, batch 4, SGD lr 0.01, momentum 0.9, weight decay 0.0005, seed 0, device cpu, threads 4
- Augmentation: random horizontal flip (p=0.5)
- Internal resize: shorter side 512 px (the base checkpoint was trained at 320)
- Best epoch by validation mAP@0.5: 5 (val mAP@0.5 0.411)
- Wall time: 6442 s; software: torch 2.15.0+cpu, torchvision 0.29.2+cpu

| Epoch | Train loss | Val mAP@0.5 |
|---|---|---|
| 1 | 0.7301 | 0.369 |
| 2 | 0.6998 | 0.351 |
| 3 | 0.6849 | 0.367 |
| 4 | 0.6482 | 0.407 |
| 5 | 0.6583 | 0.411 |

## Evaluation on the held-out test split

- mAP@0.5: **0.595**, mAP@[0.5:0.95]: **0.373** over 2 classes with ground truth
- Method: class-aware greedy one-to-one matching, 101-point interpolated AP averaged over IoU 0.50:0.95; no crowd/ignore handling; classes without ground truth are excluded, not scored zero

| Class | Ground truth | Predictions | AP@0.5 | AP@[0.5:0.95] |
|---|---|---|---|---|
| helmet | 290 | 389 | 0.855 | 0.596 |
| head | 3644 | 4211 | 0.336 | 0.151 |

Confusion matrix at score >= 0.5 and IoU >= 0.5 (rows = ground truth, columns = predictions):

| | helmet | head | background |
|---|---|---|---|
| **helmet** | 239 | 2 | 49 |
| **head** | 4 | 1048 | 2592 |
| **background** | 27 | 355 | 0 |

## ONNX export

- Exporter: torch.onnx.export (TorchScript-based, dynamo=False), traced on a labelled training image, opset 11, dynamic image size; file size 76.0 MB, SHA-256 `c968653286e8126c` (weights are not committed)
- Parity on 8 test images: max box difference 0.77537 px, max score difference 0.007331, label mismatches 0
- Latency (single image, 400x267, 4 CPU threads, 20 runs after warm-up): PyTorch eager median 174.9 ms (p95 217.1), ONNX Runtime median 67.3 ms (p95 81.0) on Intel64 Family 6 Model 154 Stepping 3, GenuineIntel
- Latency is a local measurement on one machine; it is not a benchmark of hardware.

## Licence note

TorchVision model code and the COCO-pretrained checkpoint are BSD-3-Clause licensed; original pipeline code is MIT. No AGPL component is used, so the exported model can be deployed in a client product without copyleft obligations. Confirm the client dataset's own terms separately.

## Limitations

- Scores describe this dataset's images and labelling; crowded scenes, tiny objects and a different label policy move them, so inspect per-class and per-slice results before trusting the mean.
- Scores are not calibrated probabilities; the confusion matrix threshold is a choice, not a fact.
- Greedy matching without crowd/ignore semantics differs slightly from pycocotools on crowded scenes.
- CPU training resizes the shorter side to 512 px; objects of a few pixels are lost.
- No tracking, segmentation, drift monitoring or serving layer is included.

## Before client deployment

Re-run the pipeline on representative licensed data with every object labelled, inspect per-class and per-slice results, compare the ONNX latency on the target hardware, choose thresholds with a human-review boundary, and keep the dataset fingerprint and metrics JSON with the delivered model.
