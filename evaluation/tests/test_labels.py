import json
import unittest

import numpy as np

from rare_event_eval.labels import (LabelError, LabelSpec, resolve_labels, LABEL_MISSING_TARGET,
                                    LABEL_MULTICLASS_REQUIRES_OVR, LABEL_NEGATIVE_ABSENT, LABEL_POSITIVE_ABSENT,
                                    LABEL_POSITIVE_REQUIRED, LABEL_SINGLE_CLASS, LABEL_SINGLETON_CLASS,
                                    LABEL_SPEC_INVALID, REASON_CODES)


class LabelTestCase(unittest.TestCase):
    def code_of(self, values, spec, **kwargs):
        with self.assertRaises(LabelError) as caught:
            resolve_labels(values, spec, **kwargs)
        return caught.exception.code


class BinaryLabels(LabelTestCase):
    def test_arbitrary_strings_and_explicit_positive(self):
        values = ["churned", "stayed", "stayed", "churned", "stayed"]
        result = resolve_labels(values, LabelSpec("churned", "stayed"))
        self.assertEqual(result.y.tolist(), [1, 0, 0, 1, 0])
        self.assertEqual(result.mapping["positive_label"], "churned")
        self.assertEqual(result.mapping["negative_label"], "stayed")
        self.assertEqual(result.mapping["counts"]["positive_rows"], 2)
        self.assertEqual(result.mapping["counts"]["negative_rows"], 3)
        self.assertEqual({(item["label"], item["internal"]) for item in result.mapping["original_labels"]},
                         {("churned", 1), ("stayed", 0)})

    def test_majority_class_can_be_positive_and_rarest_is_not_chosen(self):
        values = ["common"] * 8 + ["rare"] * 2
        majority = resolve_labels(values, LabelSpec("common"))
        self.assertEqual(int(majority.y.sum()), 8)
        self.assertEqual(majority.mapping["negative_label"], "rare")
        minority = resolve_labels(values, LabelSpec("rare"))
        self.assertEqual(int(minority.y.sum()), 2)

    def test_reversed_class_order_does_not_change_the_mapping(self):
        forward = resolve_labels(["a", "b", "b", "a", "b", "a"], LabelSpec("a"))
        reverse = resolve_labels(["b", "a", "a", "b", "a", "b"], LabelSpec("a"))
        self.assertEqual(forward.mapping["original_labels"], reverse.mapping["original_labels"])
        self.assertEqual(forward.y.tolist(), [1, 0, 0, 1, 0, 1])
        self.assertEqual(reverse.y.tolist(), [0, 1, 1, 0, 1, 0])

    def test_positive_is_required_and_never_inferred(self):
        for positive in (None, "", "   ", float("nan")):
            with self.subTest(positive=positive):
                self.assertEqual(self.code_of(["a", "b", "a", "b"], LabelSpec(positive)), LABEL_POSITIVE_REQUIRED)

    def test_positive_absent_and_negative_absent(self):
        self.assertEqual(self.code_of(["a", "b", "a", "b"], LabelSpec("c")), LABEL_POSITIVE_ABSENT)
        self.assertEqual(self.code_of(["a", "b", "a", "b"], LabelSpec("a", "c")), LABEL_NEGATIVE_ABSENT)

    def test_invalid_specs(self):
        data = ["a", "b", "a", "b"]
        self.assertEqual(self.code_of(data, LabelSpec("a", "a")), LABEL_SPEC_INVALID)
        self.assertEqual(self.code_of(data, LabelSpec("a", mode="multiclass")), LABEL_SPEC_INVALID)
        self.assertEqual(self.code_of(data, LabelSpec("a", "b", mode="one_vs_rest")), LABEL_SPEC_INVALID)
        self.assertEqual(self.code_of(data, LabelSpec("a"), missing_policy="impute"), LABEL_SPEC_INVALID)

    def test_nonstring_labels_are_compared_by_text_and_kept_in_the_record(self):
        result = resolve_labels(np.array([1, 0, 0, 1, 0]), LabelSpec(1))
        self.assertEqual(result.y.tolist(), [1, 0, 0, 1, 0])
        self.assertEqual(result.mapping["positive_label"], "1")
        json.dumps(result.mapping, allow_nan=False)

    def test_reason_codes_are_stable_strings(self):
        self.assertEqual(set(REASON_CODES), {
            "LABEL_POSITIVE_REQUIRED", "LABEL_SPEC_INVALID", "LABEL_MISSING_TARGET", "LABEL_SINGLE_CLASS",
            "LABEL_SINGLETON_CLASS", "LABEL_MULTICLASS_REQUIRES_OVR", "LABEL_POSITIVE_ABSENT", "LABEL_NEGATIVE_ABSENT"})

    def test_probability_orientation_is_recorded_and_internal_one_is_the_positive_label(self):
        result = resolve_labels(["x", "y", "y", "x"], LabelSpec("y"))
        self.assertEqual(result.y.tolist(), [0, 1, 1, 0])
        self.assertIn("never inverted", result.mapping["probability_orientation"])


class MulticlassAndOneVersusRest(LabelTestCase):
    values = ["red", "green", "blue", "green", "blue", "red", "blue", "green"]

    def test_three_labels_need_an_explicit_one_vs_rest_choice(self):
        self.assertEqual(self.code_of(self.values, LabelSpec("red")), LABEL_MULTICLASS_REQUIRES_OVR)
        self.assertEqual(self.code_of(self.values, LabelSpec("red", "blue")), LABEL_MULTICLASS_REQUIRES_OVR)

    def test_one_vs_rest_keeps_original_mapping_and_removes_nothing(self):
        result = resolve_labels(self.values, LabelSpec("red", mode="one_vs_rest"))
        self.assertEqual(result.y.tolist(), [1, 0, 0, 0, 0, 1, 0, 0])
        mapping = result.mapping
        self.assertEqual(mapping["mode"], "one_vs_rest")
        self.assertEqual(mapping["negative_labels"], ["blue", "green"])
        self.assertIsNone(mapping["negative_label"])
        self.assertEqual({item["label"]: item["count"] for item in mapping["original_labels"]},
                         {"blue": 3, "green": 3, "red": 2})
        self.assertEqual(mapping["counts"]["effective_rows"], len(self.values))
        self.assertEqual(result.kept_indices.tolist(), list(range(len(self.values))))

    def test_other_singleton_labels_are_merged_not_removed(self):
        result = resolve_labels(["a", "a", "b", "b", "c"], LabelSpec("a", mode="one_vs_rest"))
        self.assertEqual(result.y.tolist(), [1, 1, 0, 0, 0])
        self.assertEqual(result.mapping["counts"]["negative_rows"], 3)


class MissingSingletonAndSingleClass(LabelTestCase):
    def test_missing_targets_raise_by_default_and_are_not_a_class(self):
        for blank in (None, "", "  ", float("nan")):
            with self.subTest(blank=blank):
                with self.assertRaises(LabelError) as caught:
                    resolve_labels(["a", "b", blank, "a", "b"], LabelSpec("a"))
                self.assertEqual(caught.exception.code, LABEL_MISSING_TARGET)
                self.assertEqual(caught.exception.details["missing_count"], 1)
                self.assertEqual(caught.exception.details["first_rows"], [2])

    def test_strings_such_as_na_are_real_labels_not_missing(self):
        result = resolve_labels(["NA", "ok", "NA", "ok"], LabelSpec("NA"))
        self.assertEqual(result.mapping["counts"]["missing_target_rows"], 0)

    def test_drop_reported_lists_exclusions_with_counts_and_row_positions(self):
        values = ["a", "b", None, "a", "", "b", "a"]
        result = resolve_labels(values, LabelSpec("a"), missing_policy="drop_reported")
        self.assertEqual(result.kept_indices.tolist(), [0, 1, 3, 5, 6])
        self.assertEqual(result.y.tolist(), [1, 0, 1, 0, 1])
        self.assertEqual([item["row_index"] for item in result.exclusions], [2, 4])
        self.assertTrue(all(item["reason_code"] == LABEL_MISSING_TARGET for item in result.exclusions))
        counts = result.mapping["counts"]
        self.assertEqual((counts["input_rows"], counts["missing_target_rows"], counts["effective_rows"]), (7, 2, 5))
        self.assertEqual(result.mapping["missing_policy"], "drop_reported")
        json.dumps(result.mapping, allow_nan=False)

    def test_listed_exclusions_are_capped_but_counted(self):
        values = ["a", "b"] * 5 + [None] * 150
        result = resolve_labels(values, LabelSpec("a"), missing_policy="drop_reported")
        self.assertEqual(len(result.exclusions), 150)
        self.assertEqual(len(result.mapping["excluded_rows"]), 100)
        self.assertTrue(result.mapping["excluded_rows_truncated"])
        self.assertEqual(result.mapping["counts"]["missing_target_rows"], 150)

    def test_dropping_missing_rows_cannot_hide_a_single_class(self):
        self.assertEqual(self.code_of(["a", None, "a", None], LabelSpec("a"), missing_policy="drop_reported"),
                         LABEL_SINGLE_CLASS)

    def test_singleton_classes_raise_and_are_never_removed(self):
        self.assertEqual(self.code_of(["a"] + ["b"] * 6, LabelSpec("a")), LABEL_SINGLETON_CLASS)
        self.assertEqual(self.code_of(["a"] + ["b"] * 6, LabelSpec("b")), LABEL_SINGLETON_CLASS)
        self.assertEqual(self.code_of(["a", "b", "b", "c", "d"], LabelSpec("a", mode="one_vs_rest")),
                         LABEL_SINGLETON_CLASS)

    def test_single_class_target(self):
        self.assertEqual(self.code_of(["a"] * 5, LabelSpec("a")), LABEL_SINGLE_CLASS)
        self.assertEqual(self.code_of(["a"] * 5, LabelSpec("a", mode="one_vs_rest")), LABEL_SINGLE_CLASS)

    def test_positive_absent_takes_precedence_for_a_target_of_one_other_label(self):
        self.assertEqual(self.code_of(["b"] * 5, LabelSpec("a")), LABEL_POSITIVE_ABSENT)

    def test_error_records_are_json_safe(self):
        with self.assertRaises(LabelError) as caught:
            resolve_labels(["a"] * 4, LabelSpec("a"))
        record = caught.exception.to_record()
        self.assertEqual(record["reason_code"], LABEL_SINGLE_CLASS)
        json.dumps(record, allow_nan=False)
        self.assertIsInstance(caught.exception, ValueError)


if __name__ == "__main__":
    unittest.main()
