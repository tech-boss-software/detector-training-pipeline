import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import read_classes, require_clean, scan, split
from infer import collect_images
from model_card import render
from synthetic import CLASSES, sample_scene
from boxes import iou


def fake_size(path):
    return 100, 50


def make_dataset(root, labels):
    (root / "images").mkdir()
    (root / "labels").mkdir()
    (root / "classes.txt").write_text("a\nb\n", encoding="utf-8")
    for stem, text in labels.items():
        (root / "images" / f"{stem}.png").write_bytes(b"not really a png")
        if text is not None:
            (root / "labels" / f"{stem}.txt").write_text(text, encoding="utf-8")


class DatasetTests(unittest.TestCase):
    def test_scan_counts_objects_negatives_problems_and_orphans(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_dataset(root, {"one": "0 0.5 0.5 0.2 0.2\n1 0.25 0.5 0.1 0.4\n", "two": None, "three": "5 0.5 0.5 0.1 0.1\n\n0 0.5 0.5 0.1 0.1\n"})
            (root / "labels" / "ghost.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            report, records = scan(root, size_reader=fake_size)
            self.assertEqual(report["images"], 3)
            self.assertEqual(report["negative_images"], 1)
            self.assertEqual(report["objects"], 3)
            self.assertEqual(report["objects_per_class"], {"a": 2, "b": 1})
            self.assertEqual(len(report["problems"]), 1)
            self.assertEqual(report["problems"][0]["line"], 1)
            self.assertEqual(report["orphan_labels"], ["ghost.txt"])
            self.assertEqual(records[0]["objects"][0]["box_xyxy"], [40.0, 20.0, 60.0, 30.0])
            with self.assertRaises(ValueError):
                require_clean(report)
            report["problems"] = []
            require_clean(report)

    def test_fingerprint_changes_with_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_dataset(root, {"one": "0 0.5 0.5 0.2 0.2\n"})
            first = scan(root, size_reader=fake_size)[0]["fingerprint"]
            (root / "labels" / "one.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            self.assertNotEqual(first, scan(root, size_reader=fake_size)[0]["fingerprint"])

    def test_classes_file_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                read_classes(root)
            (root / "classes.txt").write_text("a\na\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_classes(root)

    def test_split_is_deterministic_disjoint_and_complete(self):
        records = [{"image_id": f"{i}.png"} for i in range(20)]
        first, second = split(records, seed=3), split(records, seed=3)
        self.assertEqual(first, second)
        self.assertNotEqual(first["train"], split(records, seed=4)["train"])
        parts = [set(first[k]) for k in ("train", "val", "test")]
        self.assertEqual(set.union(*parts), {r["image_id"] for r in records})
        self.assertEqual(sum(len(p) for p in parts), 20)
        self.assertEqual((len(parts[0]), len(parts[1]), len(parts[2])), (14, 3, 3))
        self.assertEqual([len(split(records[:3], 0)[k]) for k in ("train", "val", "test")], [1, 1, 1])
        with self.assertRaises(ValueError):
            split(records[:2], 0)
        with self.assertRaises(ValueError):
            split(records, 0, (0.5, 0.5, 0.5))


class SyntheticTests(unittest.TestCase):
    def test_scene_objects_are_inside_the_image_and_barely_overlap(self):
        rng = random.Random(1)
        for _ in range(50):
            objects = sample_scene(rng, size=320)
            self.assertGreaterEqual(len(objects), 1)
            for item in objects:
                self.assertIn(item["class_id"], range(len(CLASSES)))
                x0, y0, x1, y1 = item["box_xyxy"]
                self.assertTrue(0 <= x0 < x1 <= 320 and 0 <= y0 < y1 <= 320)
            for i, a in enumerate(objects):
                for b in objects[i + 1:]:
                    self.assertLessEqual(iou(a["box_xyxy"], b["box_xyxy"]), 0.15)

    def test_scene_sampling_is_seeded(self):
        self.assertEqual(sample_scene(random.Random(9)), sample_scene(random.Random(9)))


class InferAndCardTests(unittest.TestCase):
    def test_collect_images_rejects_unknown_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "a.png").write_bytes(b"x")
            (root / "b.txt").write_bytes(b"x")
            self.assertEqual([p.name for p in collect_images([str(root)])], ["a.png"])
            with self.assertRaises(ValueError):
                collect_images([str(root / "b.txt")])

    def test_model_card_renders_metrics_and_undefined_values(self):
        training = {"model": {"name": "m", "weights": "w"}, "split": {"seed": 0, "train": [1, 2], "val": [3], "test": [4]},
                    "config": {"epochs": 1, "batch_size": 2, "lr": .01, "momentum": .9, "weight_decay": 5e-4, "seed": 0, "augmentation": "flip"},
                    "environment": {"device": "cpu", "threads": 2, "torch": "t", "torchvision": "tv"},
                    "best_epoch": 1, "best_val_map50": 0.5, "seconds": 3.0, "epochs": [{"epoch": 1, "train_loss": 0.4, "val_map50": None}]}
        evaluation = {"map50": 0.75, "map50_95": None, "classes_with_ground_truth": 1, "method": "greedy",
                      "per_class": {"a": {"ground_truth": 2, "predictions": 3, "ap50": 0.75, "ap50_95": None}},
                      "confusion": {"labels": ["a", "background"], "score_threshold": .5, "iou_threshold": .5, "matrix": [[2, 0], [1, 0]]}}
        report = {"root": "r", "classes": ["a"], "images": 4, "negative_images": 0, "objects": 2, "objects_per_class": {"a": 2}, "fingerprint": "abc"}
        text = render(training, evaluation, None, report)
        self.assertIn("**0.750**", text)
        self.assertIn("undefined", text)
        self.assertNotIn("## ONNX export", text)
        self.assertIn("| **a** | 2 | 0 |", text)


if __name__ == "__main__":
    unittest.main()
