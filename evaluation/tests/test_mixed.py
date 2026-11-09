import json
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mixed_support import (NO_DASHES, assert_close, assert_finite_json, mixed_table, numeric_table, patched_runner,
                           table_to_csv, to_arrays)
from rare_event_eval import mixed as mixed_module
from rare_event_eval.core import CostAssumptions
from rare_event_eval.labels import LabelError, LabelSpec
from rare_event_eval.mixed import MixedError, fit_tabular_model, train_comparison_tabular
from rare_event_eval.mixed_models import (RecipeError, TabularPreprocessor, make_mixed_model, preprocessor_of,
                                          resample_mixed)
from rare_event_eval.registry import MIXED_REGISTRY, REGISTRY, describe_mixed_model, get_mixed_spec
from rare_event_eval.reports import render_html, training_report
from rare_event_eval.splits import make_split
from rare_event_eval.tabular import TabularError, TabularLimits
from rare_event_eval.training import MODEL_NAMES, make_model, train_comparison

try:
    import pandas
except ImportError:  # pragma: no cover - exercised in the venv without pandas
    pandas = None

FRAUD = LabelSpec("fraud", "ok")


def positives(columns, label="fraud"):
    return np.array([value == label for value in columns["label"]]).astype(int)


def run(columns, **kwargs):
    """Run the adapter with the real worker entry point in this process. Returns (result, runner)."""
    kwargs.setdefault("target", "label")
    kwargs.setdefault("label_spec", FRAUD)
    kwargs.setdefault("models", ("logistic",))
    runner, patch = patched_runner()
    with patch:
        return train_comparison_tabular(columns, **kwargs), runner


def forbid_workers():
    return mock.patch("rare_event_eval.mixed.run_worker_directory", side_effect=AssertionError("a worker was started"))


class MixedPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.columns = mixed_table()
        cls.result, cls.runner = run(cls.columns, models=("logistic", "rf_weighted"))

    def test_numeric_categorical_and_missing_values_train_successfully(self):
        result = self.result
        self.assertEqual(result["schema_version"], "training-1.1")
        for name in ("logistic", "rf_weighted"):
            row = result["outcomes"][name]
            self.assertEqual(row["status"], "success", row.get("error"))
            self.assertEqual(row["execution"]["recipe_variant"], "mixed")
            self.assertTrue(row["model_record"]["supports_mixed"])
            self.assertEqual(row["model_record"]["recipe_id"], f"rare_event_eval/{name}-mixed/1")
            self.assertEqual(row["execution"]["threads"], 1)
        self.assertEqual(result["predictors"], ["age", "income", "color", "city"])
        self.assertEqual(result["predictor_kinds"], {"age": "numeric", "income": "numeric", "color": "categorical", "city": "categorical"})
        self.assertEqual(result["excluded_columns"], [{"column": "label", "role": "target", "missing": 0, "distinct": 2,
                                                       "distinct_capped": False, "reason": "metadata column; never used as a predictor"}])
        self.assertEqual(result["preprocessing"]["adapter"]["recipe_variant"], "mixed")
        self.assertTrue(result["preprocessing"]["mixed_type_support"])
        self.assertTrue(result["preprocessing"]["record_identical_across_models"])
        self.assertFalse(result["fallback_policy"]["fallback_used"])
        self.assertEqual(result["failures"], [])
        self.assertEqual(result["input_parsing"]["rows_used"], 140)
        missing = {item["name"]: item["missing_cells"] for item in result["preprocessing"]["record"]["predictors"]}
        self.assertGreater(sum(missing["age"].values()), 0)
        self.assertGreater(sum(missing["color"].values()), 0)

    def test_json_has_no_nan_or_infinity_and_html_escapes_category_names(self):
        assert_finite_json(self, self.result)
        columns = mixed_table()
        columns["color"] = [("<script>alert(1)</script>" if i % 3 == 0 else value) for i, value in enumerate(columns["color"])]
        result, _ = run(columns)
        self.assertEqual(result["outcomes"]["logistic"]["status"], "success")
        page = render_html(training_report(result))
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", page)
        self.assertNotIn("<script>alert(1)", page)

    def test_the_outer_split_is_exact_and_built_once(self):
        split = self.result["split"]
        counts = {name: part["rows"] for name, part in split["achieved"]["partitions"].items()}
        self.assertEqual(counts, {"train": 84, "validation": 28, "test": 28})
        independent = make_split(positives(self.columns), mode="stratified", seed=42)
        self.assertEqual([len(independent.train), len(independent.validation), len(independent.test)], [84, 28, 28])
        self.assertEqual(split["requested"]["mode"], "stratified")
        self.assertFalse(split["achieved"]["fallback_used"])
        self.assertEqual(len({call["config"]["model_name"] for call in self.runner.calls}), 2)
        for call in self.runner.calls:
            self.assertEqual(call["arrays"]["train"].tolist(), independent.train.tolist())
            self.assertEqual(call["arrays"]["test"].tolist(), independent.test.tolist())

    def test_worker_input_is_typed_columns_without_objects(self):
        call = self.runner.calls[0]
        kinds = {spec["name"]: spec["kind"] for spec in call["config"]["columns"]}
        self.assertEqual(kinds, {"age": "numeric", "income": "numeric", "color": "categorical", "city": "categorical"})
        for spec in call["config"]["columns"]:
            array = call["arrays"][spec["key"]]
            self.assertEqual(array.dtype.kind, "f" if spec["kind"] == "numeric" else "U")
        for name, array in call["arrays"].items():
            self.assertIn(array.dtype.kind, "biufU", name)
        self.assertEqual(call["arrays"]["labels"].dtype.kind, "U")
        self.assertEqual(call["config"]["adapter"], "tabular")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "refused.npz"
            with self.assertRaises(TabularError):
                mixed_module._write_arrays(target, {"bad": np.array([object()], dtype=object)})
            self.assertFalse(target.exists())
            mixed_module._write_arrays(target, {"numbers": np.arange(3.0), "words": np.array(["a", "bc"])})
            with np.load(target, allow_pickle=False) as loaded:
                self.assertEqual(loaded["words"].tolist(), ["a", "bc"])

    def test_real_subprocess_run_with_a_resampling_model(self):
        result = train_comparison_tabular(mixed_table(n=100, positives=20), target="label", label_spec=FRAUD,
                                          models=("logistic", "smote_rf"), timeout_seconds=120)
        for name in ("logistic", "smote_rf"):
            row = result["outcomes"][name]
            self.assertEqual(row["status"], "success", row.get("error"))
            self.assertGreater(row["worker_wall_seconds"], 0)
        event = result["outcomes"]["smote_rf"]["execution"]["resampling_events"][0]
        self.assertEqual(event["smote_variant"], "SMOTENC")
        assert_finite_json(self, result)


class UnseenCategoriesAndAlignment(unittest.TestCase):
    def setUp(self):
        self.names = ("x", "color", "city")
        self.train = np.array([[1.0, "red", "a"], [2.0, "green", "b"], [3.0, "red", "a"], [np.nan, "", "b"],
                               [5.0, "blue", ""], [6.0, "red", ""]], dtype=object)
        self.later = np.array([[2.5, "purple", "c"], [np.nan, "", ""], [4.0, "green", "a"]], dtype=object)

    def fitted(self):
        return TabularPreprocessor(numeric_columns=(0,), categorical_columns=(1, 2), column_names=self.names).fit(self.train)

    def test_unseen_category_is_an_all_zero_block_and_nothing_is_refitted(self):
        pre = self.fitted()
        vocabulary_before = [list(item) for item in pre.vocabularies_]
        mean_before = pre.scaler_.mean_.copy()
        coded = pre.encode(self.later)
        self.assertEqual(coded[0, 1], len(pre.vocabularies_[0]))  # dedicated unseen code
        self.assertEqual(coded[0, 2], len(pre.vocabularies_[1]))
        expanded = pre.expand(coded)
        color_block = slice(1, 1 + pre.cardinalities_[0])
        self.assertTrue((expanded[0, color_block] == 0).all())
        self.assertEqual(expanded[2, color_block].sum(), 1)
        self.assertEqual([list(item) for item in pre.vocabularies_], vocabulary_before)
        np.testing.assert_array_equal(pre.scaler_.mean_, mean_before)
        np.testing.assert_array_equal(pre.transform(self.later), pre.transform(self.later))
        self.assertEqual(pre.count_unseen(self.later), {"color": 1, "city": 1})

    def test_missing_categorical_cells_take_the_most_frequent_training_value(self):
        pre = self.fitted()
        self.assertEqual(pre.modes_, ["red", "a"])  # tie between a and b for city goes to the smaller string
        coded = pre.encode(self.later)
        self.assertEqual(coded[1, 1], pre.mode_codes_[0])
        self.assertEqual(coded[1, 2], pre.mode_codes_[1])

    def test_columns_are_identical_for_every_partition(self):
        pre = self.fitted()
        widths = {pre.transform(part).shape[1] for part in (self.train, self.later, self.train[:2])}
        self.assertEqual(widths, {len(pre.feature_names_)})
        self.assertEqual(pre.feature_names_, ["x", "color=blue", "color=green", "color=red", "city=a", "city=b"])

    def test_end_to_end_unseen_categories_and_column_alignment(self):
        columns = mixed_table()
        y = positives(columns)
        split = make_split(y, mode="stratified", seed=42)
        columns["city"] = list(columns["city"])
        for index in split.test[:5]:
            columns["city"][index] = "only_in_test"
        for index in split.validation[:3]:
            columns["color"][index] = "only_in_validation"
        result, _ = run(columns)
        self.assertEqual(result["outcomes"]["logistic"]["status"], "success")
        record = result["preprocessing"]["record"]
        names = record["expanded_feature_names"]
        self.assertNotIn("city=only_in_test", names)
        self.assertNotIn("color=only_in_validation", names)
        self.assertEqual(record["categorical_columns"]["city"]["unseen_category_cells"], {"train": 0, "validation": 0, "test": 5})
        self.assertEqual(record["categorical_columns"]["color"]["unseen_category_cells"], {"train": 0, "validation": 3, "test": 0})
        # Independently derived expectation: numeric columns, then one block per categorical column in table order.
        expected = ["age", "income"]
        for column in ("color", "city"):
            values = {columns[column][i] for i in split.train if columns[column][i] != ""}
            expected += [f"{column}={value}" for value in sorted(values)]
        self.assertEqual(names, expected)
        alignment = record["alignment"]
        self.assertEqual({tuple(shape[1:]) for shape in alignment["partition_shapes"].values()}, {(len(expected),)})
        self.assertEqual([shape[0] for shape in alignment["partition_shapes"].values()], [84, 28, 28])
        self.assertTrue(alignment["identical_columns_across_partitions"])
        self.assertEqual(alignment["columns_sha256"], mixed_module.sha256(json.dumps(expected).encode()).hexdigest())


class LeakageBoundaries(unittest.TestCase):
    def test_no_train_fitted_transform_sees_validation_or_test_rows(self):
        columns = mixed_table(with_row_id=True)
        split = make_split(positives(columns), mode="stratified", seed=42)
        train_ids = set(split.train.tolist())
        seen = {"preprocessor": [], "imputer": [], "scaler": [], "logistic": [], "forest": []}

        def spy(cls, key, first_column_ids=True):
            original = cls.fit

            def wrapper(self, X, *args, **kwargs):
                X = np.asarray(X)
                seen[key].append(X[:, 0].astype(float).astype(int).tolist() if first_column_ids else len(X))
                return original(self, X, *args, **kwargs)
            return mock.patch.object(cls, "fit", wrapper)

        with spy(TabularPreprocessor, "preprocessor"), spy(SimpleImputer, "imputer"), spy(StandardScaler, "scaler"), \
                spy(LogisticRegression, "logistic", False):
            result, _ = run(columns)
        self.assertEqual(result["outcomes"]["logistic"]["status"], "success")
        for key in ("preprocessor", "imputer", "scaler"):
            self.assertGreaterEqual(len(seen[key]), 2, key)  # the recording fit and the model pipeline fit
            for ids in seen[key]:
                self.assertEqual(set(ids), train_ids, key)
        self.assertEqual(seen["logistic"], [len(split.train)])

    def test_resampling_and_forest_fits_see_only_training_sized_input(self):
        columns = mixed_table(with_row_id=True)
        split = make_split(positives(columns), mode="stratified", seed=42)
        sizes = []
        original = RandomForestClassifier.fit

        def wrapper(self, X, *args, **kwargs):
            sizes.append(len(X))
            return original(self, X, *args, **kwargs)
        with mock.patch.object(RandomForestClassifier, "fit", wrapper):
            result, _ = run(columns, models=("rf_raw",))
        self.assertEqual(result["outcomes"]["rf_raw"]["status"], "success")
        self.assertEqual(sizes, [len(split.train)])

    def test_preprocessing_statistics_do_not_depend_on_other_partitions(self):
        columns = mixed_table(with_row_id=True)
        split = make_split(positives(columns), mode="stratified", seed=42)
        changed = {key: list(values) for key, values in columns.items()}
        for index in np.concatenate([split.validation, split.test]):
            changed["income"][index] = "100000"
            changed["city"][index] = "zzz"
        first, _ = run(columns)
        second, _ = run(changed)
        for key in ("expanded_feature_names", "numeric_columns"):
            self.assertEqual(first["preprocessing"]["record"][key], second["preprocessing"]["record"][key])
        self.assertNotIn("city=zzz", second["preprocessing"]["record"]["expanded_feature_names"])
        train_missing = lambda result: [item["missing_cells"]["train"] for item in result["preprocessing"]["record"]["predictors"]]
        self.assertEqual(train_missing(first), train_missing(second))
        self.assertEqual(second["preprocessing"]["record"]["categorical_columns"]["city"]["unseen_category_cells"]["test"],
                         len(split.test))


class Splits(unittest.TestCase):
    def test_group_split_keeps_groups_disjoint_and_excludes_the_group_column(self):
        columns = mixed_table(n=160, positives=40, with_group=True)
        result, runner = run(columns, split_mode="group", group_column="entity")
        self.assertEqual(result["outcomes"]["logistic"]["status"], "success")
        arrays = runner.calls[0]["arrays"]
        groups = arrays["groups"]
        sets = [set(groups[arrays[name]].tolist()) for name in ("train", "validation", "test")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(sum(len(arrays[name]) for name in ("train", "validation", "test")), 160)
        self.assertNotIn("entity", result["predictors"])
        role = {item["column"]: item for item in result["excluded_columns"]}["entity"]
        self.assertEqual((role["role"], role["used_by_split"]), ("group", True))
        self.assertTrue(runner.calls[0]["config"]["split_mode"] == "group")

    def test_feature_group_split_never_separates_identical_predictor_patterns(self):
        base = mixed_table(n=90, positives=24)
        columns = {name: values + (values[:40] if name != "label" else base["label"][:40]) for name, values in base.items()}
        result, runner = run(columns, split_mode="feature_group")
        self.assertEqual(result["outcomes"]["logistic"]["status"], "success", result["outcomes"]["logistic"].get("error"))
        arrays = runner.calls[0]["arrays"]
        part = np.empty(130, dtype=int)
        for number, name in enumerate(("train", "validation", "test")):
            part[arrays[name]] = number
        patterns = {}
        for row in range(130):
            patterns.setdefault(tuple(columns[name][row] for name in ("age", "income", "color", "city")), set()).add(int(part[row]))
        self.assertTrue(all(len(parts) == 1 for parts in patterns.values()))
        self.assertLess(len(patterns), 130)  # duplicates exist
        diagnostics = result["outcomes"]["logistic"]["execution"]["split"]
        self.assertEqual(diagnostics["test_rows_matching_training_features"], 0)
        self.assertTrue(diagnostics["group_overlap_checked"])

    def test_chronological_split_keeps_equal_timestamps_together(self):
        columns = mixed_table(n=120, with_time=True, time_ties=5, positive_rows=range(0, 120, 6))
        result, runner = run(columns, split_mode="chronological", time_column="when")
        self.assertEqual(result["outcomes"]["logistic"]["status"], "success", result["outcomes"]["logistic"].get("error"))
        arrays = runner.calls[0]["arrays"]
        times = arrays["times"]
        spans = [(times[arrays[name]].min(), times[arrays[name]].max()) for name in ("train", "validation", "test")]
        self.assertLess(spans[0][1], spans[1][0])
        self.assertLess(spans[1][1], spans[2][0])
        sets = [set(times[arrays[name]].tolist()) for name in ("train", "validation", "test")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        counts = {name: part["rows"] for name, part in result["split"]["achieved"]["partitions"].items()}
        self.assertEqual(counts, {"train": 75, "validation": 25, "test": 20})  # 72 and 96 were moved to tie boundaries
        self.assertNotIn("when", result["predictors"])

    def test_infeasible_requests_fail_before_any_worker_and_name_the_request(self):
        few_groups = mixed_table(n=60, positives=15, with_group=True, group_size=30)
        with forbid_workers():
            with self.assertRaises(MixedError) as raised:
                train_comparison_tabular(few_groups, target="label", label_spec=FRAUD, split_mode="group", group_column="entity")
            self.assertEqual(raised.exception.code, "SPLIT_INFEASIBLE")
            self.assertEqual(raised.exception.details["requested_mode"], "group")
            late_positives = mixed_table(n=60, with_time=True, positive_rows=range(50, 60))
            with self.assertRaises(MixedError) as raised:
                train_comparison_tabular(late_positives, target="label", label_spec=FRAUD, split_mode="chronological", time_column="when")
            self.assertEqual(raised.exception.code, "SPLIT_INFEASIBLE")
            with self.assertRaises(MixedError) as raised:  # three rare rows cannot fill three partitions
                train_comparison_tabular(mixed_table(n=80, positives=3), target="label", label_spec=FRAUD)
            self.assertEqual(raised.exception.code, "SPLIT_INFEASIBLE")

    def test_unsupported_and_incompatible_combinations_are_typed_errors(self):
        columns = mixed_table(n=120, with_time=True, with_group=True, positive_rows=range(0, 120, 6))
        cases = [
            (dict(split_mode="chronological", time_column="when", group_column="entity"), "SPLIT_UNSUPPORTED"),
            (dict(split_mode="chronological", time_column="when", calibration="sigmoid"), "CALIBRATION_INCOMPATIBLE"),
            (dict(split_mode="group"), "OPTION_INVALID"),
            (dict(split_mode="chronological"), "OPTION_INVALID"),
            (dict(split_mode="random"), "OPTION_INVALID"),
            (dict(fallback_policy="random_over"), "OPTION_INVALID"),
            (dict(calibration="isotonic"), "OPTION_INVALID"),
        ]
        with forbid_workers():
            for options, code in cases:
                with self.subTest(options=options), self.assertRaises(MixedError) as raised:
                    train_comparison_tabular(columns, target="label", label_spec=FRAUD, **options)
                self.assertEqual(raised.exception.code, code)

    def test_missing_group_and_unparseable_time_values_are_rejected(self):
        columns = mixed_table(n=120, with_time=True, with_group=True, positive_rows=range(0, 120, 6))
        columns["entity"][7] = ""
        columns["when"][9] = "not a date"
        with forbid_workers():
            with self.assertRaises(MixedError) as raised:
                train_comparison_tabular(columns, target="label", label_spec=FRAUD, split_mode="group", group_column="entity")
            self.assertEqual(raised.exception.code, "GROUP_MISSING")
            with self.assertRaises(MixedError) as raised:
                train_comparison_tabular(columns, target="label", label_spec=FRAUD, split_mode="chronological", time_column="when")
            self.assertEqual(raised.exception.code, "TIME_INVALID")


class BoundedExpansion(unittest.TestCase):
    def test_high_cardinality_is_rejected_before_any_fit(self):
        columns = mixed_table()
        columns["big"] = [f"v{i % 60}" for i in range(140)]
        with forbid_workers(), mock.patch.object(TabularPreprocessor, "fit", side_effect=AssertionError("fit was reached")):
            with self.assertRaises(TabularError) as raised:
                train_comparison_tabular(columns, target="label", label_spec=FRAUD, models=("logistic",))
        self.assertEqual(raised.exception.code, "TABLE_CARDINALITY_EXCEEDED")
        self.assertEqual(raised.exception.details["violations"][0]["column"], "big")

    def test_expanded_width_limit_is_checked_before_any_fit(self):
        with forbid_workers():
            with self.assertRaises(TabularError) as raised:
                train_comparison_tabular(mixed_table(), target="label", label_spec=FRAUD, limits=TabularLimits(max_expanded_features=6))
        self.assertEqual(raised.exception.code, "TABLE_EXPANSION_EXCEEDED")
        self.assertEqual(raised.exception.details["violations"][0]["estimated_width"], 9)

    def test_excluding_a_column_by_role_works_and_is_recorded_not_silent(self):
        columns = mixed_table()
        columns["big"] = [f"v{i % 60}" for i in range(140)]
        result, _ = run(columns, id_columns=("big",))
        self.assertEqual(result["outcomes"]["logistic"]["status"], "success")
        self.assertNotIn("big", result["predictors"])
        self.assertIn(("big", "id"), {(item["column"], item["role"]) for item in result["excluded_columns"]})

    def test_the_worker_checks_the_actual_expanded_width_before_building_a_model(self):
        _, runner = run(mixed_table())
        call = runner.calls[0]
        config = dict(call["config"], limits=dict(call["config"]["limits"], max_expanded_features=3))
        with mock.patch.object(LogisticRegression, "fit", side_effect=AssertionError("a model was fitted")):
            with self.assertRaises(RecipeError) as raised:
                fit_tabular_model(call["arrays"], config)
        self.assertEqual(raised.exception.code, "EXPANSION_EXCEEDED")

    def test_predictor_selection_and_exclusion_are_recorded(self):
        columns = mixed_table(with_row_id=True)
        result, _ = run(columns, predictors=["age", "color"])
        self.assertEqual(result["predictors"], ["age", "color"])
        roles = {item["column"]: item["role"] for item in result["excluded_columns"]}
        self.assertEqual((roles["income"], roles["city"], roles["row_id"]), ("not_selected",) * 3)
        result, _ = run(columns, exclude_columns=["row_id", "city"])
        self.assertEqual(result["predictors"], ["age", "income", "color"])
        self.assertEqual({item["column"]: item["role"] for item in result["excluded_columns"]}["row_id"], "excluded_by_user")
        for options in (dict(predictors=["nope"]), dict(predictors=["label"]), dict(exclude_columns=["nope"]),
                        dict(predictors=["age"], exclude_columns=["age"])):
            with self.subTest(options=options), forbid_workers(), self.assertRaises(MixedError) as raised:
                train_comparison_tabular(columns, target="label", label_spec=FRAUD, **options)
            self.assertEqual(raised.exception.code, "PREDICTOR_UNKNOWN")


class CategoricalAwareResampling(unittest.TestCase):
    def coded(self, n_minority=14):
        rng = np.random.RandomState(0)
        n = 100
        y = np.zeros(n, dtype=int)
        y[:n_minority] = 1
        rows = np.empty((n, 3), dtype=object)
        rows[:, 0] = rng.normal(size=n) + y
        rows[:, 1] = np.where(rng.rand(n) < .3 + .4 * y, "red", rng.choice(["green", "blue"], n))
        rows[:, 2] = rng.choice(["a", "b", "c", "d", "e"], n)
        pre = TabularPreprocessor(numeric_columns=(0,), categorical_columns=(1, 2), column_names=("x", "color", "city")).fit(rows)
        return pre, pre.encode(rows), y

    def test_synthetic_category_codes_are_members_of_the_learned_vocabulary(self):
        pre, coded, y = self.coded()
        for method in ("smote", "clustered_smote"):
            with self.subTest(method=method):
                Xr, yr, event = resample_mixed(coded, y, pre, method=method, retention=.5, seed=7)
                retained = int(sum(event["retained_counts"])) if method == "clustered_smote" else len(y)
                synthetic = Xr[retained:, pre.n_numeric_:]
                self.assertEqual(len(synthetic), event["synthetic_rows"])
                self.assertGreater(len(synthetic), 0)
                self.assertTrue((synthetic == np.round(synthetic)).all())
                for offset, size in enumerate(pre.cardinalities_):
                    self.assertTrue(((synthetic[:, offset] >= 0) & (synthetic[:, offset] < size)).all())
                self.assertEqual(event["smote_variant"], "SMOTENC")
                self.assertTrue(event["synthetic_category_codes_in_vocabulary"])
                self.assertEqual(sorted(event["output_counts"]), sorted(np.bincount(yr).tolist()))
                self.assertEqual(event["output_counts"][0], event["output_counts"][1])

    def test_smotenc_is_used_with_the_code_columns_declared_categorical(self):
        from imblearn.over_sampling import SMOTENC
        pre, coded, y = self.coded()
        with mock.patch("imblearn.over_sampling.SMOTENC", wraps=SMOTENC) as spy:
            resample_mixed(coded, y, pre, method="smote", retention=.5, seed=7)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.call_args.kwargs["categorical_features"], [1, 2])
        self.assertEqual(spy.call_args.kwargs["k_neighbors"], 5)

    def test_a_fractional_category_code_is_rejected_not_rounded(self):
        from imblearn.over_sampling import SMOTENC
        pre, coded, y = self.coded()
        real = SMOTENC.fit_resample

        def fractional(self, X, y):
            Xr, yr = real(self, X, y)
            Xr = Xr.copy()
            Xr[-1, 1] += 0.5
            return Xr, yr
        with mock.patch.object(SMOTENC, "fit_resample", fractional):
            with self.assertRaises(RecipeError) as raised:
                resample_mixed(coded, y, pre, method="smote", retention=.5, seed=7)
        self.assertEqual(raised.exception.code, "RESAMPLING_INVALID_CATEGORY_CODE")

    def test_clustering_uses_the_documented_expanded_representation(self):
        pre, coded, y = self.coded()
        seen = []
        from sklearn.cluster import MiniBatchKMeans
        original = MiniBatchKMeans.fit_predict

        def spy(self, X, *args, **kwargs):
            seen.append(np.asarray(X))
            return original(self, X, *args, **kwargs)
        with mock.patch.object(MiniBatchKMeans, "fit_predict", spy):
            resample_mixed(coded, y, pre, method="clustered", retention=.5, seed=7)
        self.assertEqual(seen[0].shape, (int((y == 0).sum()), pre.expanded_width_))
        self.assertTrue(set(np.unique(seen[0][:, 1:])) <= {0.0, 1.0})  # indicators, not ordinal codes

    def test_all_categorical_predictors_fail_the_resampling_recipes_with_a_reason_code(self):
        columns = mixed_table()
        for name in ("age", "income"):
            del columns[name]
        result, _ = run(columns, models=("logistic", "smote_rf", "erf_one_bag"))
        self.assertEqual(result["outcomes"]["logistic"]["status"], "success")
        for name in ("smote_rf", "erf_one_bag"):
            row = result["outcomes"][name]
            self.assertEqual((row["status"], row["reason_code"]), ("failed", "RESAMPLING_NO_NUMERIC_FEATURES"))
            self.assertNotIn("report", row)
        self.assertEqual(sorted(item["model"] for item in result["failures"]), ["erf_one_bag", "smote_rf"])


class RareClassFailures(unittest.TestCase):
    def test_a_single_training_minority_row_fails_resampling_recipes_without_fallback(self):
        columns = mixed_table(n=60, with_time=True, time_ties=3, positive_rows=[10, 40, 55])
        split = make_split(positives(columns), mode="chronological", seed=42,
                           times=np.arange(60) // 3)
        self.assertEqual(int(positives(columns)[split.train].sum()), 1)
        models = ("logistic", "smote_rf", "csrboost", "erf_one_bag", "erf_no_smote")
        result, _ = run(columns, split_mode="chronological", time_column="when", models=models)
        outcomes = result["outcomes"]
        self.assertEqual(outcomes["logistic"]["status"], "success")
        self.assertEqual(outcomes["erf_no_smote"]["status"], "success")  # no SMOTE step, so nothing is infeasible
        for name in ("smote_rf", "csrboost", "erf_one_bag"):
            self.assertEqual((outcomes[name]["status"], outcomes[name]["reason_code"]), ("failed", "RESAMPLING_INFEASIBLE"), name)
            self.assertIn("no fallback", outcomes[name]["error"])
            self.assertNotIn("report", outcomes[name])
        self.assertEqual({item["model"] for item in result["failures"]}, {"smote_rf", "csrboost", "erf_one_bag"})
        self.assertEqual(result["fallback_policy"], {"policy": "strict", "fallback_used": False,
                                                     "description": result["fallback_policy"]["description"]})
        self.assertIn("RESAMPLING_INFEASIBLE", json.dumps(result))
        assert_finite_json(self, result)

    def test_two_training_minority_rows_are_the_smallest_feasible_case(self):
        columns = mixed_table(n=80, positives=4)
        result, _ = run(columns, models=("smote_rf", "csrboost"))
        for name in ("smote_rf", "csrboost"):
            row = result["outcomes"][name]
            self.assertEqual(row["status"], "success", row.get("error"))
            self.assertEqual(row["execution"]["resampling_events"][0]["smote_neighbors"], 1)

    def test_failures_do_not_hide_a_comparison_in_the_html_report(self):
        columns = mixed_table(n=60, with_time=True, time_ties=3, positive_rows=[10, 40, 55])
        result, _ = run(columns, split_mode="chronological", time_column="when", models=("logistic", "smote_rf"))
        page = render_html(training_report(result))
        self.assertIn("smote_rf", page)
        self.assertIn("were not silently removed", page)


class Calibration(unittest.TestCase):
    def test_sigmoid_calibration_runs_inside_training_on_mixed_inputs(self):
        columns = mixed_table(n=150, positives=45)
        result, _ = run(columns, calibration="sigmoid")
        row = result["outcomes"]["logistic"]
        self.assertEqual(row["status"], "success", row.get("error"))
        self.assertEqual(row["execution"]["calibration"], "sigmoid")
        self.assertIn("calibration", {step["step"] for step in row["execution"]["constructed_steps"]})
        self.assertFalse(row["execution"]["threshold_selection"]["test_labels_used"])

    def test_infeasible_inner_folds_fail_the_request_before_any_worker(self):
        with forbid_workers(), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # scikit-learn warns that the rare class is smaller than the fold count
            with self.assertRaises(MixedError) as raised:
                train_comparison_tabular(mixed_table(n=80, positives=4), target="label", label_spec=FRAUD, calibration="sigmoid")
        self.assertEqual(raised.exception.code, "CALIBRATION_INFEASIBLE")

    def test_the_worker_also_refuses_chronological_sigmoid(self):
        columns = mixed_table(n=120, with_time=True, time_ties=5, positive_rows=range(0, 120, 6))
        _, runner = run(columns, split_mode="chronological", time_column="when")
        call = runner.calls[0]
        with self.assertRaises(RecipeError) as raised:
            fit_tabular_model(call["arrays"], dict(call["config"], calibration="sigmoid"))
        self.assertEqual(raised.exception.code, "CALIBRATION_INCOMPATIBLE")


class Determinism(unittest.TestCase):
    def test_same_seed_gives_identical_results_and_another_seed_a_different_split(self):
        columns = mixed_table()
        models = ("logistic", "rf_weighted", "erf_one_bag")
        first, _ = run(columns, models=models)
        second, _ = run(columns, models=models)
        for name in models:
            self.assertEqual(first["outcomes"][name]["status"], "success")
            assert_close(self, first["outcomes"][name]["report"]["models"], second["outcomes"][name]["report"]["models"], 0.0)
            assert_close(self, first["outcomes"][name]["execution"]["threshold_selection"],
                         second["outcomes"][name]["execution"]["threshold_selection"], 0.0)
        self.assertEqual(first["split"], second["split"])
        other, _ = run(columns, models=("logistic",), seed=7)
        self.assertNotEqual(first["split"]["achieved"]["partitions"]["test"]["indices_sha256"],
                            other["split"]["achieved"]["partitions"]["test"]["indices_sha256"])


class LabelSpecsThroughTheAdapter(unittest.TestCase):
    def test_arbitrary_string_labels_and_a_majority_positive_choice(self):
        columns = mixed_table(labels=("Zebra case", "alpha-case"))
        result, _ = run(columns, label_spec=LabelSpec("alpha-case", "Zebra case"))
        mapping = result["label_mapping"]
        self.assertEqual((mapping["positive_label"], mapping["negative_label"]), ("alpha-case", "Zebra case"))
        self.assertEqual(mapping["counts"]["positive_rows"], 112)
        report = result["outcomes"]["logistic"]["report"]
        self.assertEqual(report["task"]["positive_label"], "alpha-case")
        self.assertGreater(report["data"]["positive_prevalence"], 0.7)
        self.assertEqual(result["outcomes"]["logistic"]["status"], "success")

    def test_reversed_order_gives_the_complementary_mapping(self):
        columns = mixed_table()
        forward, _ = run(columns, label_spec=LabelSpec("fraud", "ok"))
        reverse, _ = run(columns, label_spec=LabelSpec("ok", "fraud"))
        self.assertEqual(forward["label_mapping"]["counts"]["positive_rows"], 28)
        self.assertEqual(reverse["label_mapping"]["counts"]["positive_rows"], 112)
        self.assertEqual(reverse["label_mapping"]["internal_encoding"], {"1": "positive_label", "0": "negative_labels"})
        self.assertEqual(reverse["outcomes"]["logistic"]["report"]["task"]["negative_label"], "fraud")

    def test_one_vs_rest_keeps_the_original_to_binary_mapping(self):
        columns = mixed_table()
        columns["label"] = [("b" if value == "fraud" else ("a" if i % 2 else "c")) for i, value in enumerate(columns["label"])]
        with self.assertRaises(LabelError) as raised:
            train_comparison_tabular(columns, target="label", label_spec=LabelSpec("b", "a"))
        self.assertEqual(raised.exception.code, "LABEL_MULTICLASS_REQUIRES_OVR")
        with self.assertRaises(LabelError) as raised:
            train_comparison_tabular(columns, target="label", label_spec=LabelSpec("b"))
        self.assertEqual(raised.exception.code, "LABEL_MULTICLASS_REQUIRES_OVR")
        result, _ = run(columns, label_spec=LabelSpec("b", mode="one_vs_rest"))
        mapping = result["label_mapping"]
        self.assertEqual(mapping["mode"], "one_vs_rest")
        self.assertEqual(mapping["negative_labels"], ["a", "c"])
        self.assertEqual({item["label"]: item["internal"] for item in mapping["original_labels"]}, {"a": 0, "b": 1, "c": 0})
        self.assertEqual(result["outcomes"]["logistic"]["report"]["task"]["negative_label"], "(all other labels)")
        self.assertEqual(result["outcomes"]["logistic"]["status"], "success")

    def test_missing_targets_are_rejected_or_excluded_with_a_record(self):
        columns = mixed_table()
        for row in (3, 50, 77):
            columns["label"][row] = ""
        with forbid_workers(), self.assertRaises(LabelError) as raised:
            train_comparison_tabular(columns, target="label", label_spec=FRAUD)
        self.assertEqual(raised.exception.code, "LABEL_MISSING_TARGET")
        result, runner = run(columns, missing_policy="drop_reported")
        self.assertEqual(result["input_parsing"]["rows_input"], 140)
        self.assertEqual(result["input_parsing"]["rows_used"], 137)
        self.assertEqual([item["row_index"] for item in result["input_parsing"]["row_exclusions"]], [3, 50, 77])
        self.assertEqual(len(runner.calls[0]["arrays"]["labels"]), 137)
        self.assertEqual(result["split"]["achieved"]["rows"], 137)

    def test_positive_label_must_be_explicit_and_present(self):
        columns = mixed_table()
        for spec in (LabelSpec(None, "ok"), LabelSpec("absent", "ok")):
            with self.subTest(spec=spec), forbid_workers(), self.assertRaises(LabelError):
                train_comparison_tabular(columns, target="label", label_spec=spec)


class NumericParityWithTheNumericPath(unittest.TestCase):
    """The adapter on a numeric-only table must equal the 0.3.0 numeric path for the same seed (tolerance 1e-12)."""

    MODELS = ("logistic", "rf_raw", "smote_rf", "csrboost", "erf_one_bag")

    def compare(self, columns, models, **options):
        X, labels = to_arrays(columns)
        old = train_comparison(X, labels, models=models, positive_label="event", negative_label="clear", seed=42,
                               timeout_seconds=120, **options)
        new, _ = run(columns, label_spec=LabelSpec("event", "clear"), models=models, seed=42, **options)
        for name in models:
            self.assertEqual(old["outcomes"][name]["status"], "success", old["outcomes"][name])
            self.assertEqual(new["outcomes"][name]["status"], "success", new["outcomes"][name])
            self.assertEqual(new["outcomes"][name]["execution"]["recipe_variant"], "numeric")
            self.assertEqual(new["outcomes"][name]["model_record"], old["outcomes"][name]["model_record"])
            assert_close(self, old["outcomes"][name]["report"]["models"], new["outcomes"][name]["report"]["models"], 1e-12,
                         f"{name}.models")
            assert_close(self, old["outcomes"][name]["execution"]["threshold_selection"],
                         new["outcomes"][name]["execution"]["threshold_selection"], 1e-12, f"{name}.threshold")
            for key in ("rows", "positive_rows", "positive_prevalence"):
                self.assertEqual(old["outcomes"][name]["report"]["data"][key], new["outcomes"][name]["report"]["data"][key])
            self.assertEqual(old["outcomes"][name]["execution"]["split"], new["outcomes"][name]["execution"]["split"])
            self.assertEqual(old["outcomes"][name]["execution"]["resampling_events"],
                             new["outcomes"][name]["execution"]["resampling_events"])
        self.assertEqual(old["split"]["achieved"]["partitions"], new["split"]["achieved"]["partitions"])
        self.assertEqual(new["preprocessing"]["adapter"]["recipe_variant"], "numeric")
        return old, new

    def test_stratified_split_with_missing_constant_and_empty_columns(self):
        self.compare(numeric_table(), self.MODELS)

    def test_feature_group_split(self):
        columns = numeric_table(n=110, positives=30)
        columns = {name: values + (values[:30] if name != "label" else values[:30]) for name, values in columns.items()}
        self.compare(columns, ("logistic",), split_mode="feature_group")

    def test_sigmoid_calibration(self):
        self.compare(numeric_table(n=180, positives=60), ("logistic",), calibration="sigmoid")


class RecipesAndRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.columns = mixed_table(n=120, positives=24)
        cls.result, _ = run(cls.columns, models=MODEL_NAMES)

    def test_every_mixed_recipe_fits_and_records_its_own_recipe_id(self):
        resampling = {"smote_rf", "csrboost", "erf", "erf_no_clustering", "erf_no_smote", "erf_one_bag"}
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                row = self.result["outcomes"][name]
                self.assertEqual(row["status"], "success", row.get("error"))
                self.assertEqual(row["execution"]["recipe_variant"], "mixed")
                self.assertEqual(row["model_record"]["recipe_id"], MIXED_REGISTRY[f"{name}-mixed"].recipe_id)
                self.assertTrue(row["model_record"]["supports_mixed"])
                self.assertEqual(row["model_record"]["variant_of"], name)
                events = row["execution"]["resampling_events"]
                self.assertEqual(bool(events), name in resampling)
                for event in events:
                    if "smote" in event["method"]:
                        self.assertEqual(event["smote_variant"], "SMOTENC")
                        self.assertTrue(event["synthetic_category_codes_in_vocabulary"])
                    else:
                        self.assertNotIn("smote_variant", event)
        self.assertEqual(len(self.result["outcomes"]["erf"]["execution"]["resampling_events"]), 20)
        self.assertEqual(len(self.result["outcomes"]["erf_one_bag"]["execution"]["resampling_events"]), 1)
        self.assertEqual(self.result["failures"], [])
        assert_finite_json(self, self.result)

    def test_mixed_registry_is_separate_complete_and_leaves_numeric_ids_alone(self):
        self.assertEqual(set(MIXED_REGISTRY), {f"{name}-mixed" for name in MODEL_NAMES})
        recipes = [spec.recipe_id for spec in MIXED_REGISTRY.values()]
        self.assertEqual(len(set(recipes)), len(recipes))
        self.assertFalse(set(recipes) & {spec.recipe_id for spec in REGISTRY.values()})
        for model_id, spec in MIXED_REGISTRY.items():
            self.assertTrue(spec.supports_mixed)
            self.assertEqual(spec.scope, "package_mixed")
            self.assertEqual(spec.variant_of + "-mixed", model_id)
            self.assertIs(get_mixed_spec(spec.variant_of), spec)
        self.assertFalse(any(spec.supports_mixed for spec in REGISTRY.values()))
        self.assertEqual(REGISTRY["erf"].recipe_id, "rare_event_eval/erf/1")
        with self.assertRaises(ValueError):
            get_mixed_spec("legacy_rf_weighted_200")

    def test_declared_mixed_parameters_equal_the_built_objects(self):
        for model_id, spec in MIXED_REGISTRY.items():
            base = spec.variant_of
            with self.subTest(model=model_id):
                for seed in (42, 7):
                    described = describe_mixed_model(base, seed=seed)
                    declared = spec.resolved_params(seed)
                    self.assertEqual(described["spec"]["params"], declared)
                    built = described["estimator"]
                    self.assertEqual(declared["estimator"]["class"], built["class"])
                    for key, value in declared["estimator"].items():
                        if key != "class":
                            self.assertEqual(built["params"][key], value, f"{model_id}.{key}")

    def test_mixed_estimators_share_capacity_with_the_numeric_recipes(self):
        for name in MODEL_NAMES:
            numeric = make_model(name, seed=11, minority_count=10).named_steps["model"]
            mixed = make_mixed_model(name, seed=11, minority_count=10, numeric_columns=(0,), categorical_columns=(1,),
                                     column_names=("a", "b")).named_steps["model"]
            for key, value in numeric.get_params(deep=False).items():
                if key in mixed.get_params(deep=False):
                    self.assertEqual(mixed.get_params(deep=False)[key], value, f"{name}.{key}")

    def test_new_modules_have_no_en_or_em_dashes(self):
        root = Path(__file__).resolve().parents[1] / "src" / "rare_event_eval"
        for name in ("mixed.py", "mixed_models.py", "registry.py", "cli.py"):
            self.assertIsNone(NO_DASHES.search((root / name).read_text(encoding="utf-8")), name)


@unittest.skipUnless(pandas, "pandas is absent")
class DataFrameInput(unittest.TestCase):
    def test_a_dataframe_with_nan_and_category_dtype_trains(self):
        columns = mixed_table()
        frame = pandas.DataFrame({name: (values if name == "label" else
                                         [np.nan if value == "" else (float(value) if name in ("age", "income") else value)
                                          for value in values])
                                  for name, values in columns.items()})
        frame["city"] = frame["city"].astype("category")
        result, _ = run(frame)
        row = result["outcomes"]["logistic"]
        self.assertEqual(row["status"], "success", row.get("error"))
        self.assertEqual(result["predictor_kinds"]["city"], "categorical")
        self.assertEqual(result["predictor_kinds"]["age"], "numeric")
        self.assertEqual(result["input_parsing"]["parser"], "table_object")
        assert_finite_json(self, result)

    def test_csv_bytes_and_dataframe_give_the_same_metrics(self):
        columns = mixed_table()
        from_csv, _ = run(table_to_csv(columns))
        from_frame, _ = run(pandas.read_csv(__import__("io").BytesIO(table_to_csv(columns)), keep_default_na=False, na_values=[""]))
        assert_close(self, from_csv["outcomes"]["logistic"]["report"]["models"],
                     from_frame["outcomes"]["logistic"]["report"]["models"], 1e-12)


class CsvBytesInput(unittest.TestCase):
    def test_csv_bytes_are_parsed_with_limits_and_hash_recorded(self):
        raw = table_to_csv(mixed_table())
        result, _ = run(raw)
        self.assertEqual(result["input_parsing"]["parser"], "parse_table_csv")
        self.assertEqual(result["input_sha256"], result["input_parsing"]["details"]["input_sha256"])
        with forbid_workers(), self.assertRaises(TabularError) as raised:
            train_comparison_tabular(raw, target="label", label_spec=FRAUD, limits=TabularLimits(max_rows=50))
        self.assertEqual(raised.exception.code, "TABLE_TOO_MANY_ROWS")

    def test_pandas_gate_is_optional_and_typed(self):
        columns = mixed_table()
        with mock.patch("rare_event_eval.mixed._import_pandas", side_effect=ImportError("no pandas")):
            with forbid_workers(), self.assertRaises(MixedError) as raised:
                train_comparison_tabular(columns, target="label", label_spec=FRAUD, require_pandas_for_categorical=True)
            self.assertEqual(raised.exception.code, "PANDAS_REQUIRED")
            self.assertIn("pandas", str(raised.exception))
            result, _ = run(numeric_table(), label_spec=LabelSpec("event", "clear"), require_pandas_for_categorical=True)
            self.assertEqual(result["outcomes"]["logistic"]["status"], "success")  # numeric-only tables never need it
            result, _ = run(columns)  # the library itself does not need pandas
            self.assertEqual(result["outcomes"]["logistic"]["status"], "success")

    @unittest.skipIf(pandas, "pandas is installed")
    def test_gate_with_pandas_really_absent(self):
        with forbid_workers(), self.assertRaises(MixedError) as raised:
            train_comparison_tabular(mixed_table(), target="label", label_spec=FRAUD, require_pandas_for_categorical=True)
        self.assertEqual(raised.exception.code, "PANDAS_REQUIRED")

    def test_categorical_handling_reject_refuses_categorical_predictors(self):
        with forbid_workers(), self.assertRaises(MixedError) as raised:
            train_comparison_tabular(mixed_table(), target="label", label_spec=FRAUD, categorical_handling="reject")
        self.assertEqual(raised.exception.code, "CATEGORICAL_REJECTED")


class CostsAndDescription(unittest.TestCase):
    def test_costs_and_description_reach_the_report(self):
        result, _ = run(mixed_table(), costs=CostAssumptions(1.0, 10.0, 0.1), description="Fictional mixed table.")
        report = result["outcomes"]["logistic"]["report"]
        self.assertEqual(report["policies"]["cost_assumptions"]["false_negative_cost"], 10.0)
        self.assertEqual(report["data"]["description"], "Fictional mixed table. Fresh fit with separate evaluation partitions.")
        selection = result["outcomes"]["logistic"]["execution"]["threshold_selection"]
        self.assertEqual(selection["objective"], "minimize_assumed_cost")
        with forbid_workers(), self.assertRaises(MixedError):
            train_comparison_tabular(mixed_table(), target="label", label_spec=FRAUD, description="x" * 451)


if __name__ == "__main__":
    unittest.main()
