from __future__ import annotations

import unittest

from zhaw_at_touche.evaluation_utils import compute_metrics, counts_from_pairs


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


if __name__ == "__main__":
    unittest.main()
