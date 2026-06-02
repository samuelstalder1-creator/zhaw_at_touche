from __future__ import annotations

import unittest

from zhaw_at_touche.evaluation_utils import (
    compact_metrics_summary,
    compute_metrics,
    counts_from_pairs,
    metrics_dict,
    validation_metrics_payload,
)


class EvaluationUtilsTests(unittest.TestCase):
    def test_counts_and_metrics(self) -> None:
        counts, labels, total = counts_from_pairs([1, 1, 0, 0], [1, 0, 0, 0])
        per_label, macro, weighted = compute_metrics(counts, labels)

        self.assertEqual(total, 4)
        self.assertEqual(labels, [0, 1])
        self.assertEqual(counts[(1, 1)], 1)
        self.assertEqual(counts[(1, 0)], 1)
        self.assertEqual(counts[(0, 0)], 2)
        self.assertEqual(len(per_label), 2)
        self.assertAlmostEqual(macro["f1"], 0.7333333333, places=6)
        self.assertAlmostEqual(weighted["f1"], 0.7333333333, places=6)

    def test_validation_metrics_payload_includes_monitoring_fields(self) -> None:
        summary = metrics_dict([0, 0, 1, 1], [0, 1, 0, 1])

        metrics = validation_metrics_payload(loss=0.25, summary=summary)

        self.assertAlmostEqual(metrics["loss"], 0.25)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["samples"], 4)

    def test_compact_metrics_summary_removes_macro_and_weighted_fields(self) -> None:
        summary = metrics_dict([0, 0, 1, 1], [0, 1, 0, 1])

        compact = compact_metrics_summary(summary)

        self.assertEqual(
            compact["confusion_matrix"],
            {
                "true_negative": 1,
                "false_positive": 1,
                "false_negative": 1,
                "true_positive": 1,
            },
        )
        self.assertAlmostEqual(compact["accuracy"], 0.5)
        self.assertAlmostEqual(compact["precision"], 0.5)
        self.assertAlmostEqual(compact["recall"], 0.5)
        self.assertAlmostEqual(compact["f1"], 0.5)
        self.assertNotIn("macro", compact)
        self.assertNotIn("weighted", compact)
        self.assertNotIn("positive_label", compact)


if __name__ == "__main__":
    unittest.main()
