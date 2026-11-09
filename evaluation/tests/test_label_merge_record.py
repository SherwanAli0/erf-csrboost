"""One-vs-rest: the report records every other label merged into the negative class, singletons included.

The mapping is recorded in ``label_mapping`` of the training-1.1 result (and so in the provenance block of the
HTML report) on both the numeric path and the mixed-type adapter. Nothing is dropped and nothing is merged
without a record. These tests do not change behavior; they pin the record.
"""
import json
import unittest

import numpy as np

from mixed_support import mixed_table, numeric_table, patched_runner, to_arrays
from rare_event_eval.labels import LabelSpec, resolve_labels
from rare_event_eval.mixed import train_comparison_tabular
from rare_event_eval.reports import render_html, training_report
from rare_event_eval.training import train_comparison

# "solo_alpha" and "solo_beta" occur once each; "other" occurs several times (all are negative labels).


def relabelled(labels, positive):
    """Replace some negative labels so the table has two singleton labels and one repeated other label."""
    labels = np.asarray(labels, dtype=object).copy()
    negatives = [i for i, value in enumerate(labels) if value != positive]
    labels[negatives[0]] = "solo_alpha"
    labels[negatives[1]] = "solo_beta"
    for index in negatives[2:12]:
        labels[index] = "other"
    return labels


def assert_merge_record(test, mapping, labels, positive):
    counts = {name: int(sum(1 for value in labels if value == name)) for name in set(labels)}
    test.assertEqual(counts["solo_alpha"], 1)
    test.assertEqual(counts["solo_beta"], 1)
    test.assertEqual(mapping["mode"], "one_vs_rest")
    test.assertEqual(mapping["positive_label"], positive)
    test.assertIsNone(mapping["negative_label"])
    test.assertEqual(mapping["negative_labels"], sorted(name for name in counts if name != positive))
    recorded = {item["label"]: (item["count"], item["internal"]) for item in mapping["original_labels"]}
    test.assertEqual(recorded, {name: (counts[name], 1 if name == positive else 0) for name in counts})
    test.assertEqual(recorded["solo_alpha"], (1, 0))  # singleton labels are listed, not removed or hidden
    test.assertEqual(recorded["solo_beta"], (1, 0))
    merged_rows = sum(count for name, (count, internal) in recorded.items() if internal == 0)
    test.assertEqual(mapping["counts"]["negative_rows"], merged_rows)
    test.assertEqual(mapping["counts"]["effective_rows"], len(labels))  # no row was removed
    test.assertEqual(mapping["counts"]["missing_target_rows"], 0)
    test.assertEqual(mapping["internal_encoding"], {"1": "positive_label", "0": "negative_labels"})
    json.dumps(mapping, allow_nan=False)


class ResolverRecord(unittest.TestCase):
    def test_singletons_are_in_the_record_and_the_rows_are_kept(self):
        values = ["event"] * 4 + ["clear"] * 6 + ["other"] * 3 + ["solo_alpha", "solo_beta"]
        result = resolve_labels(values, LabelSpec("event", mode="one_vs_rest"))
        assert_merge_record(self, result.mapping, values, "event")
        self.assertEqual(result.y.tolist(), [1] * 4 + [0] * 11)
        self.assertEqual(result.kept_indices.tolist(), list(range(len(values))))


class NumericPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        columns = numeric_table()
        X, labels = to_arrays(columns)
        cls.labels = relabelled(labels, "event")
        cls.result = train_comparison(X, cls.labels, models=("logistic",), label_spec=LabelSpec("event", mode="one_vs_rest"),
                                      timeout_seconds=120)

    def test_mapping_lists_every_merged_label_with_its_count(self):
        assert_merge_record(self, self.result["label_mapping"], list(self.labels), "event")
        self.assertEqual(self.result["outcomes"]["logistic"]["status"], "success")
        self.assertEqual(self.result["input_parsing"]["rows_excluded"], 0)

    def test_the_provenance_block_and_html_carry_the_same_record(self):
        report = training_report(self.result)
        recorded = {item["label"]: item["count"] for item in report["training_provenance"]["label_mapping"]["original_labels"]}
        self.assertEqual(recorded["solo_alpha"], 1)
        self.assertEqual(recorded["solo_beta"], 1)
        page = render_html(report)
        for name in ("solo_alpha", "solo_beta", "other"):
            self.assertIn(name, page)


class MixedAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        columns = mixed_table()
        cls.labels = relabelled(columns["label"], "fraud")
        columns["label"] = [str(value) for value in cls.labels]
        runner, patch = patched_runner()
        with patch:
            cls.result = train_comparison_tabular(columns, target="label", label_spec=LabelSpec("fraud", mode="one_vs_rest"),
                                                  models=("logistic",))

    def test_mapping_lists_every_merged_label_with_its_count(self):
        assert_merge_record(self, self.result["label_mapping"], list(self.labels), "fraud")
        self.assertEqual(self.result["outcomes"]["logistic"]["status"], "success")
        self.assertEqual(self.result["input_parsing"]["rows_excluded"], 0)

    def test_the_html_report_carries_the_record(self):
        page = render_html(training_report(self.result))
        for name in ("solo_alpha", "solo_beta", "other"):
            self.assertIn(name, page)


if __name__ == "__main__":
    unittest.main()
