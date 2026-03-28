from __future__ import annotations

import unittest

from zhaw_at_touche.overlap_utils import RowRef, collect_overlap_report, dataset_sizes


class OverlapUtilsTests(unittest.TestCase):
    def test_collect_overlap_report_for_query(self) -> None:
        split_rows = {
            "train": [
                RowRef(row_id="train-1", query="shared query", response="alpha"),
                RowRef(row_id="train-2", query="train only", response="shared response"),
            ],
            "validation": [
                RowRef(row_id="validation-1", query="shared query", response="beta"),
                RowRef(row_id="validation-2", query="validation only", response="shared response"),
            ],
            "test": [
                RowRef(row_id="test-1", query="shared query", response="gamma"),
                RowRef(row_id="test-2", query="test only", response="shared response"),
            ],
        }

        self.assertEqual(dataset_sizes(split_rows), {"train": 2, "validation": 2, "test": 2})

        comparisons = collect_overlap_report("query", split_rows, sample_limit=1)
        counts = {comparison.label: comparison.overlap_count for comparison in comparisons}

        self.assertEqual(counts["train vs validation"], 1)
        self.assertEqual(counts["train vs test"], 1)
        self.assertEqual(counts["validation vs test"], 1)
        self.assertEqual(counts["train vs validation vs test"], 1)
        self.assertEqual(comparisons[0].samples[0].key_text, "shared query")

    def test_collect_overlap_report_for_response(self) -> None:
        split_rows = {
            "train": [RowRef(row_id="train-1", query="a", response="shared response")],
            "validation": [RowRef(row_id="validation-1", query="b", response="shared response")],
            "test": [RowRef(row_id="test-1", query="c", response="shared response")],
        }

        comparisons = collect_overlap_report("response", split_rows, sample_limit=2)
        counts = {comparison.label: comparison.overlap_count for comparison in comparisons}

        self.assertEqual(counts["train vs validation"], 1)
        self.assertEqual(counts["train vs test"], 1)
        self.assertEqual(counts["validation vs test"], 1)
        self.assertEqual(counts["train vs validation vs test"], 1)


if __name__ == "__main__":
    unittest.main()
