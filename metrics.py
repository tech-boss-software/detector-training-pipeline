"""Dependency-free detection metrics: COCO-style AP, PR curves and a confusion matrix.

Records are flat dicts: {"image_id": str, "class_id": int, "box_xyxy": [...]} and
predictions additionally carry "score". Crowd/ignore semantics are not implemented;
every ground-truth object is expected to be labelled.
"""
from __future__ import annotations

import math

from boxes import iou, validate_box

IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
RECALL_POINTS = [i / 100 for i in range(101)]


def _validate(records, need_score):
    for row in records:
        validate_box(row["box_xyxy"])
        if not isinstance(row.get("image_id"), str) or not row["image_id"]:
            raise ValueError("image_id must be a nonempty string")
        class_id = row.get("class_id")
        if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 0:
            raise ValueError("class_id must be a nonnegative integer")
        if need_score:
            score = float(row["score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("score must be finite and between 0 and 1")


def match_class(predictions, ground_truth, class_id, iou_threshold):
    """Greedy confidence-ordered one-to-one matching for one class.

    Returns (tp_flags in score order, positives). Each ground-truth box is
    matched at most once; a second overlapping prediction is a false positive.
    """
    truths = {}
    for index, row in enumerate(ground_truth):
        if row["class_id"] == class_id:
            truths.setdefault(row["image_id"], []).append((index, row["box_xyxy"]))
    positives = sum(len(v) for v in truths.values())
    matched = set()
    flags = []
    ordered = sorted((row for row in predictions if row["class_id"] == class_id), key=lambda r: -float(r["score"]))
    for row in ordered:
        best, best_iou = None, 0.0
        for index, box in truths.get(row["image_id"], []):
            if index in matched:
                continue
            overlap = iou(row["box_xyxy"], box)
            if overlap > best_iou:
                best, best_iou = index, overlap
        if best is not None and best_iou >= iou_threshold:
            matched.add(best)
            flags.append(True)
        else:
            flags.append(False)
    return flags, positives


def precision_recall(flags, positives):
    tp = fp = 0
    precision, recall = [], []
    for flag in flags:
        tp += flag
        fp += not flag
        precision.append(tp / (tp + fp))
        recall.append(tp / positives if positives else 0.0)
    return precision, recall


def interpolated_ap(precision, recall):
    """COCO-style 101-point interpolation with a monotone precision envelope."""
    if not precision:
        return 0.0
    envelope = list(precision)
    for i in range(len(envelope) - 2, -1, -1):
        envelope[i] = max(envelope[i], envelope[i + 1])
    total = 0.0
    for point in RECALL_POINTS:
        value = 0.0
        for p, r in zip(envelope, recall):
            if r >= point:
                value = p
                break
        total += value
    return total / len(RECALL_POINTS)


def average_precision(predictions, ground_truth, class_id, iou_threshold):
    """AP for one class; None when the class has no ground truth (undefined, not zero)."""
    flags, positives = match_class(predictions, ground_truth, class_id, iou_threshold)
    if positives == 0:
        return None
    precision, recall = precision_recall(flags, positives)
    return interpolated_ap(precision, recall)


def evaluate(predictions, ground_truth, class_names, confusion_score_threshold=0.5):
    """mAP@0.5, mAP@[0.5:0.95], per-class AP and PR points, plus a confusion matrix."""
    _validate(predictions, True)
    _validate(ground_truth, False)
    for row in [*predictions, *ground_truth]:
        if row["class_id"] >= len(class_names):
            raise ValueError("class_id outside the declared class list")
    per_class = {}
    for class_id, name in enumerate(class_names):
        aps = {}
        for threshold in IOU_THRESHOLDS:
            aps[str(threshold)] = average_precision(predictions, ground_truth, class_id, threshold)
        flags, positives = match_class(predictions, ground_truth, class_id, 0.5)
        precision, recall = precision_recall(flags, positives)
        defined = [v for v in aps.values() if v is not None]
        per_class[name] = {
            "ground_truth": positives,
            "predictions": len(flags),
            "ap50": aps["0.5"],
            "ap50_95": sum(defined) / len(defined) if defined else None,
            "ap_by_iou": aps,
            # Thin the curve so JSON stays small while keeping the shape.
            "pr_curve_iou50": [[round(r, 4), round(p, 4)] for p, r in _thin(precision, recall)],
        }
    defined50 = [v["ap50"] for v in per_class.values() if v["ap50"] is not None]
    defined5095 = [v["ap50_95"] for v in per_class.values() if v["ap50_95"] is not None]
    return {
        "map50": sum(defined50) / len(defined50) if defined50 else None,
        "map50_95": sum(defined5095) / len(defined5095) if defined5095 else None,
        "classes_with_ground_truth": len(defined50),
        "per_class": per_class,
        "confusion": confusion_matrix(predictions, ground_truth, class_names, confusion_score_threshold, 0.5),
        "method": "class-aware greedy one-to-one matching, 101-point interpolated AP averaged over IoU 0.50:0.95; "
                  "no crowd/ignore handling; classes without ground truth are excluded, not scored zero",
    }


def _thin(precision, recall, limit=200):
    pairs = list(zip(precision, recall))
    if len(pairs) <= limit:
        return pairs
    step = len(pairs) / limit
    return [pairs[int(i * step)] for i in range(limit)] + [pairs[-1]]


def confusion_matrix(predictions, ground_truth, class_names, score_threshold, iou_threshold):
    """Rows are ground truth, columns are predictions; the last index is background.

    Matching is class-agnostic by IoU so that a box with the right location but the
    wrong label lands in the off-diagonal cell instead of two background cells.
    """
    size = len(class_names) + 1
    background = len(class_names)
    matrix = [[0] * size for _ in range(size)]
    by_image = {}
    for index, row in enumerate(ground_truth):
        by_image.setdefault(row["image_id"], []).append(index)
    matched = set()
    kept = sorted((row for row in predictions if float(row["score"]) >= score_threshold), key=lambda r: -float(r["score"]))
    for row in kept:
        best, best_iou = None, 0.0
        for index in by_image.get(row["image_id"], []):
            if index in matched:
                continue
            overlap = iou(row["box_xyxy"], ground_truth[index]["box_xyxy"])
            if overlap > best_iou:
                best, best_iou = index, overlap
        if best is not None and best_iou >= iou_threshold:
            matched.add(best)
            matrix[ground_truth[best]["class_id"]][row["class_id"]] += 1
        else:
            matrix[background][row["class_id"]] += 1
    for index, row in enumerate(ground_truth):
        if index not in matched:
            matrix[row["class_id"]][background] += 1
    return {"labels": [*class_names, "background"], "score_threshold": score_threshold,
            "iou_threshold": iou_threshold, "matrix": matrix,
            "note": "rows = ground truth, columns = predictions; background row = false positives, background column = misses"}
