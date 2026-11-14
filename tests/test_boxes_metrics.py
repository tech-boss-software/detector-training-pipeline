import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boxes import clip_box, iou, parse_yolo_line, to_yolo_line, validate_box
from metrics import average_precision, confusion_matrix, evaluate, interpolated_ap, match_class


def gt(image, cls, box):
    return {"image_id": image, "class_id": cls, "box_xyxy": box}


def pred(image, cls, box, score):
    return {**gt(image, cls, box), "score": score}


class BoxTests(unittest.TestCase):
    def test_validate_and_clip(self):
        self.assertEqual(validate_box([0, 0, 2, 2]), [0.0, 0.0, 2.0, 2.0])
        for bad in ([2, 0, 1, 3], [0, 0, float("nan"), 3], [0, 0, 1]):
            with self.assertRaises(ValueError):
                validate_box(bad)
        self.assertEqual(clip_box([-5, -5, 3, 3], 10, 10), [0.0, 0.0, 3.0, 3.0])
        self.assertIsNone(clip_box([20, 20, 30, 30], 10, 10))

    def test_iou_values(self):
        self.assertAlmostEqual(iou([0, 0, 2, 2], [1, 1, 3, 3]), 1 / 7)
        self.assertEqual(iou([0, 0, 1, 1], [1, 1, 2, 2]), 0)
        self.assertEqual(iou([0, 0, 2, 2], [0, 0, 2, 2]), 1)

    def test_yolo_roundtrip_and_validation(self):
        class_id, box = parse_yolo_line("1 0.5 0.5 0.25 0.5", 200, 100, 3)
        self.assertEqual(class_id, 1)
        self.assertEqual(box, [75.0, 25.0, 125.0, 75.0])
        line = to_yolo_line(class_id, box, 200, 100)
        self.assertEqual(parse_yolo_line(line, 200, 100, 3)[1], box)
        # Tool rounding that pushes a box slightly outside is clipped, not rejected.
        self.assertEqual(parse_yolo_line("0 0.0 0.5 0.1 0.2", 100, 100, 1)[1], [0.0, 40.0, 5.0, 60.0])
        for bad in ("3 0.5 0.5 0.1 0.1", "0 0.5 0.5 0 0.1", "0 1.5 0.5 0.1 0.1", "0 0.5 0.5 0.1", "a 0.5 0.5 0.1 0.1"):
            with self.assertRaises(ValueError):
                parse_yolo_line(bad, 100, 100, 3)


class MetricTests(unittest.TestCase):
    def test_perfect_predictions_give_ap_one(self):
        truth = [gt("a", 0, [0, 0, 10, 10]), gt("b", 0, [5, 5, 20, 20])]
        preds = [pred("a", 0, [0, 0, 10, 10], .9), pred("b", 0, [5, 5, 20, 20], .8)]
        self.assertEqual(average_precision(preds, truth, 0, 0.5), 1.0)
        self.assertEqual(average_precision(preds, truth, 0, 0.95), 1.0)

    def test_duplicate_and_wrong_class_are_false_positives(self):
        truth = [gt("a", 0, [0, 0, 10, 10])]
        preds = [pred("a", 0, [0, 0, 10, 10], .9), pred("a", 0, [0, 0, 10, 10], .8), pred("a", 1, [0, 0, 10, 10], .7)]
        flags, positives = match_class(preds, truth, 0, 0.5)
        self.assertEqual((flags, positives), ([True, False], 1))
        self.assertEqual(average_precision(preds, truth, 0, 0.5), 1.0)  # TP ranked first
        self.assertIsNone(average_precision(preds, truth, 1, 0.5))  # no ground truth for class 1

    def test_lower_ranked_true_positive_reduces_ap(self):
        truth = [gt("a", 0, [0, 0, 10, 10]), gt("a", 0, [50, 50, 60, 60])]
        preds = [pred("a", 0, [100, 100, 110, 110], .95), pred("a", 0, [0, 0, 10, 10], .9), pred("a", 0, [50, 50, 60, 60], .8)]
        ap = average_precision(preds, truth, 0, 0.5)
        self.assertGreater(ap, 0.6)
        self.assertLess(ap, 0.7)

    def test_interpolation_uses_precision_envelope(self):
        self.assertAlmostEqual(interpolated_ap([1.0, 0.5, 0.667], [0.5, 0.5, 1.0]), (51 * 1.0 + 50 * 0.667) / 101, places=6)
        self.assertEqual(interpolated_ap([], []), 0.0)

    def test_evaluate_aggregates_and_reports_undefined_classes(self):
        truth = [gt("a", 0, [0, 0, 10, 10]), gt("a", 1, [20, 20, 30, 30])]
        preds = [pred("a", 0, [0, 0, 10, 10], .9), pred("a", 1, [20, 20, 30, 30], .4)]
        result = evaluate(preds, truth, ["x", "y", "z"], confusion_score_threshold=0.5)
        self.assertEqual(result["map50"], 1.0)
        self.assertEqual(result["classes_with_ground_truth"], 2)
        self.assertIsNone(result["per_class"]["z"]["ap50"])
        matrix = result["confusion"]["matrix"]
        self.assertEqual(matrix[0][0], 1)  # x matched
        self.assertEqual(matrix[1][3], 1)  # y missed because its score is under the threshold
        self.assertEqual(sum(map(sum, matrix)), 2)

    def test_confusion_matrix_cells(self):
        classes = ["x", "y"]
        truth = [gt("a", 0, [0, 0, 10, 10]), gt("a", 1, [20, 20, 30, 30]), gt("b", 0, [0, 0, 10, 10])]
        preds = [pred("a", 1, [0, 0, 10, 10], .9),      # right place, wrong label -> row x, col y
                 pred("a", 1, [20, 20, 30, 30], .9),    # correct y
                 pred("b", 0, [50, 50, 60, 60], .9)]    # false positive -> background row
        matrix = confusion_matrix(preds, truth, classes, 0.5, 0.5)["matrix"]
        self.assertEqual(matrix, [[0, 1, 1], [0, 1, 0], [1, 0, 0]])

    def test_invalid_records_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([pred("a", 0, [0, 0, 1, 1], 1.5)], [], ["x"])
        with self.assertRaises(ValueError):
            evaluate([], [gt("", 0, [0, 0, 1, 1])], ["x"])
        with self.assertRaises(ValueError):
            evaluate([], [gt("a", 5, [0, 0, 1, 1])], ["x"])
        self.assertIsNone(evaluate([], [], ["x"])["map50"])


if __name__ == "__main__":
    unittest.main()


class ParityMatchingTests(unittest.TestCase):
    def test_swapped_order_is_not_an_error_but_a_missing_partner_is(self):
        from export import match_outputs

        a = ([10, 10, 50, 50], 1, 0.91)
        b = ([200, 20, 260, 90], 1, 0.90)
        c = ([5, 5, 30, 30], 2, 0.40)
        result = match_outputs([a, b, c], [b, a, c])
        self.assertEqual(result["max_box_abs_diff"], 0.0)
        self.assertEqual(result["unmatched"], 0)
        moved = ([10, 10, 50, 52.5], 1, 0.905)
        result = match_outputs([a, b, c], [b, moved])
        self.assertAlmostEqual(result["max_box_abs_diff"], 2.5)
        self.assertAlmostEqual(result["max_score_abs_diff"], 0.005)
        self.assertEqual(result["unmatched"], 1)
        self.assertEqual(result["count_difference"], 1)
