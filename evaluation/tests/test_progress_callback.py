"""The optional ``progress`` observer of ``train_comparison_tabular``.

It reports ``splitting``, ``model_started``, ``model_finished`` and ``reporting`` events. With ``progress=None``
the result must equal a call that does not mention it, and an observer must never change or stop a run.
"""
import json
import unittest
from unittest import mock

from mixed_support import mixed_table, patched_runner
from rare_event_eval.labels import LabelSpec
from rare_event_eval.mixed import MixedError, train_comparison_tabular

VOLATILE = {"created_utc", "fit_seconds", "fit_and_evaluation_seconds", "peak_process_rss_bytes", "worker_wall_seconds"}
FRAUD = LabelSpec("fraud", "ok")


def stable(value):
    """The result without wall-clock stamps and memory measurements, which differ between any two runs."""
    if isinstance(value, dict):
        return {key: stable(item) for key, item in value.items() if key not in VOLATILE}
    if isinstance(value, list):
        return [stable(item) for item in value]
    return value


def run(columns, **kwargs):
    kwargs.setdefault("target", "label")
    kwargs.setdefault("label_spec", FRAUD)
    kwargs.setdefault("models", ("logistic", "rf_weighted"))
    _, patch = patched_runner()
    with patch:
        return train_comparison_tabular(columns, **kwargs)


class ProgressCallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.columns = mixed_table()
        cls.reference = run(cls.columns)

    def test_without_the_argument_or_with_none_the_result_is_identical(self):
        explicit = run(self.columns, progress=None)
        self.assertEqual(json.dumps(stable(explicit), sort_keys=True), json.dumps(stable(self.reference), sort_keys=True))

    def test_events_arrive_in_order_with_small_json_safe_payloads(self):
        events = []
        result = run(self.columns, progress=events.append)
        self.assertEqual([item["event"] for item in events],
                         ["splitting", "model_started", "model_finished", "model_started", "model_finished", "reporting"])
        self.assertEqual([(item["model"], item["index"], item["total"]) for item in events[1:5]],
                         [("logistic", 1, 2), ("logistic", 1, 2), ("rf_weighted", 2, 2), ("rf_weighted", 2, 2)])
        self.assertEqual([item["status"] for item in events if item["event"] == "model_finished"], ["success", "success"])
        for item in events:
            self.assertLess(len(json.dumps(item, allow_nan=False)), 200)
        self.assertEqual(json.dumps(stable(result), sort_keys=True), json.dumps(stable(self.reference), sort_keys=True))

    def test_failed_models_report_their_reason_code(self):
        columns = mixed_table()
        for name in ("age", "income"):
            del columns[name]
        events = []
        result = run(columns, models=("logistic", "smote_rf"), progress=events.append)
        finished = {item["model"]: item for item in events if item["event"] == "model_finished"}
        self.assertEqual(finished["logistic"]["status"], "success")
        self.assertNotIn("reason_code", finished["logistic"])
        self.assertEqual((finished["smote_rf"]["status"], finished["smote_rf"]["reason_code"]),
                         ("failed", "RESAMPLING_NO_NUMERIC_FEATURES"))
        self.assertEqual(result["outcomes"]["smote_rf"]["reason_code"], "RESAMPLING_NO_NUMERIC_FEATURES")

    def test_a_worker_timeout_is_reported_as_timeout_with_its_code(self):
        events = []
        timed_out = {"status": "timeout", "model": "logistic", "timeout_seconds": 1, "worker_wall_seconds": 1.0}
        with mock.patch("rare_event_eval.mixed.run_worker_directory", return_value=timed_out):
            result = train_comparison_tabular(self.columns, target="label", label_spec=FRAUD, models=("logistic",),
                                              timeout_seconds=1, progress=events.append)
        finished = [item for item in events if item["event"] == "model_finished"]
        self.assertEqual([(item["status"], item["reason_code"]) for item in finished], [("timeout", "MODEL_TIMEOUT")])
        self.assertEqual(result["outcomes"]["logistic"]["status"], "timeout")

    def test_an_observer_that_raises_cannot_change_or_stop_the_run(self):
        calls = []

        def broken(event):
            calls.append(event["event"])
            raise RuntimeError("observer failure")

        result = run(self.columns, progress=broken)
        self.assertEqual(len(calls), 6)
        self.assertEqual(json.dumps(stable(result), sort_keys=True), json.dumps(stable(self.reference), sort_keys=True))

    def test_a_progress_value_that_is_not_callable_is_refused_before_any_work(self):
        with mock.patch("rare_event_eval.mixed.run_worker_directory", side_effect=AssertionError("a worker was started")):
            with self.assertRaises(MixedError) as caught:
                train_comparison_tabular(self.columns, target="label", label_spec=FRAUD, progress="not callable")
        self.assertEqual(caught.exception.code, "OPTION_INVALID")

    def test_a_whole_request_error_emits_no_model_event(self):
        events = []
        with mock.patch("rare_event_eval.mixed.run_worker_directory", side_effect=AssertionError("a worker was started")):
            with self.assertRaises(ValueError):
                train_comparison_tabular(self.columns, target="missing_column", label_spec=FRAUD, progress=events.append)
        self.assertFalse([item for item in events if item["event"].startswith("model_")])


if __name__ == "__main__":
    unittest.main()
