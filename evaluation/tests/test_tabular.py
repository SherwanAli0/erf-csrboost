import json
import subprocess
import sys
import unittest

from rare_event_eval import tabular
from rare_event_eval.tabular import (ParsedTable, TabularError, TabularLimits, excluded_metadata_records,
                                     parse_table_csv, profile_table, table_to_dataframe)

try:
    import pandas
except ImportError:  # the optional "app" extra is not installed
    pandas = None


def code_of(test, call, *args, **kwargs):
    with test.assertRaises(TabularError) as caught:
        call(*args, **kwargs)
    return caught.exception


CSV = (b"id,entity,when,amount,color,flag,note,outcome\n"
       b"r1,e1,2026-01-01,1.5,red,x,same,event\n"
       b"r2,e1,2026-01-02,2.5,blue,x,same,clear\n"
       b"r3,e2,2026-01-03,,red,x,same,clear\n"
       b"r4,e3,2026-01-04,2.5,green,x,same,clear\n"
       b"r5,e3,2026-01-05,1.5,red,x,same,event\n")


class ProfileColumns(unittest.TestCase):
    def profile(self, **overrides):
        parsed = parse_table_csv(CSV)
        options = dict(target="outcome", group_column="entity", time_column="when", id_columns=("id",))
        options.update(overrides)
        return profile_table(parsed, **options)

    def test_target_group_time_and_ids_are_excluded_and_never_predictors(self):
        profile = self.profile()
        self.assertEqual(profile["predictors"], ["amount", "color", "flag", "note"])
        self.assertEqual({item["column"]: item["role"] for item in profile["excluded_columns"]},
                         {"id": "id", "entity": "group", "when": "time", "outcome": "target"})
        for name in ("id", "entity", "when", "outcome"):
            self.assertNotIn(name, profile["predictors"])
            self.assertIsNone(profile["columns"][name]["kind"])
            self.assertEqual(profile["columns"][name]["expanded_width"], 0)

    def test_kinds_missingness_and_constants_without_dropping_anything(self):
        profile = self.profile()
        kinds = {name: profile["columns"][name]["kind"] for name in profile["predictors"]}
        self.assertEqual(kinds, {"amount": "numeric", "color": "categorical", "flag": "constant", "note": "constant"})
        self.assertEqual(profile["constant_columns"], ["flag", "note"])
        self.assertEqual(profile["columns"]["amount"]["missing"], 1)
        self.assertAlmostEqual(profile["columns"]["amount"]["missing_fraction"], 0.2)
        self.assertEqual(profile["predictor_count"], 4)
        self.assertEqual(profile["rows"], 5)
        self.assertIn("No predictor column or row was removed.", profile["notes"])

    def test_without_declared_metadata_every_other_column_is_a_predictor(self):
        profile = profile_table(parse_table_csv(CSV), target="outcome")
        self.assertEqual(len(profile["predictors"]), 7)
        self.assertEqual([item["column"] for item in profile["excluded_columns"]], ["outcome"])

    def test_empty_column_and_duplicate_columns_and_duplicate_patterns(self):
        rows = [{"a": 1, "b": 1, "c": "", "d": "u", "y": "p"}, {"a": 2, "b": 2, "c": "", "d": "v", "y": "n"},
                {"a": 1, "b": 1, "c": "", "d": "u", "y": "n"}, {"a": 3, "b": 3, "c": "", "d": "w", "y": "p"}]
        profile = profile_table(rows, target="y")
        self.assertEqual(profile["empty_columns"], ["c"])
        self.assertEqual(profile["columns"]["c"]["kind"], "empty")
        self.assertEqual(profile["duplicate_columns"], [["a", "b"]])
        groups = profile["duplicate_feature_groups"]
        self.assertEqual((groups["groups"], groups["rows_in_groups"]), (1, 2))
        self.assertIn("not automatic proof of leakage", groups["interpretation"])

    def test_expanded_width_estimate(self):
        profile = self.profile()
        estimate = profile["expanded_feature_estimate"]
        self.assertEqual(profile["columns"]["color"]["expanded_width"], 3)
        self.assertEqual((estimate["numeric_like"], estimate["categorical_one_hot"], estimate["total"]), (3, 3, 6))

    def test_target_summary_reports_counts_and_missing(self):
        profile = self.profile()
        self.assertEqual(profile["target_summary"]["value_counts"], {"clear": 3, "event": 2})
        rows = [{"x": 1, "y": "a"}, {"x": 2, "y": ""}, {"x": 3, "y": "b"}]
        summary = profile_table(rows, target="y")["target_summary"]
        self.assertEqual((summary["missing"], summary["present"], summary["distinct"]), (1, 2, 2))

    def test_profile_is_json_safe_and_records_limits_and_parsing(self):
        profile = self.profile()
        json.dumps(profile, allow_nan=False)
        self.assertEqual(profile["limits"], tabular.asdict(TabularLimits()))
        self.assertTrue(profile["limits_checked"])
        self.assertEqual(profile["parsing"]["delimiter"]["delimiter"], ",")

    def test_role_conflicts_missing_columns_and_no_predictors(self):
        table = parse_table_csv(CSV)
        self.assertEqual(code_of(self, profile_table, table, target="outcome", group_column="outcome").code,
                         tabular.TABLE_COLUMN_ROLE_CONFLICT)
        self.assertEqual(code_of(self, profile_table, table, target="nope").code, tabular.TABLE_COLUMN_MISSING)
        self.assertEqual(code_of(self, profile_table, [{"a": 1, "y": "p"}, {"a": 2, "y": "n"}], target="y",
                                 id_columns=("a",)).code, tabular.TABLE_NO_PREDICTORS)

    def test_excluded_metadata_records_helper(self):
        records = excluded_metadata_records("y", "g", "t", ("i",))
        self.assertEqual([item["role"] for item in records], ["target", "group", "time", "id"])
        with self.assertRaises(TabularError):
            excluded_metadata_records("y", "y")

    @unittest.skipIf(pandas is None, "pandas is not installed (optional app extra)")
    def test_dataframe_input_and_lazy_conversion(self):
        frame = pandas.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": list("xyxy"), "y": list("pnpn")})
        profile = profile_table(frame, target="y")
        self.assertEqual(profile["predictors"], ["a", "b"])
        self.assertEqual(table_to_dataframe(parse_table_csv(CSV)).shape, (5, 8))

    @unittest.skipIf(pandas is not None, "only meaningful when pandas is absent")
    def test_missing_pandas_gives_a_clear_error_only_when_conversion_is_requested(self):
        with self.assertRaisesRegex(ImportError, r"\[app\]"):
            table_to_dataframe(parse_table_csv(CSV))


class LimitsRaiseBeforeHeavyWork(unittest.TestCase):
    def test_default_limits_are_documented_local_defaults(self):
        limits = TabularLimits()
        self.assertEqual((limits.max_bytes, limits.max_rows, limits.max_columns,
                          limits.max_categorical_cardinality, limits.max_expanded_features),
                         (5_000_000, 20_000, 205, 50, 400))
        self.assertIn("not verified limits of", tabular.__doc__)
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(TabularError):
                TabularLimits(max_rows=bad)

    def test_byte_limit_is_checked_before_decoding(self):
        raw = b"\xff\xfe" * 100  # would fail to decode if decoding were attempted
        error = code_of(self, parse_table_csv, raw, TabularLimits(max_bytes=50))
        self.assertEqual(error.code, tabular.TABLE_TOO_LARGE_BYTES)

    def test_column_limit_is_checked_before_any_row_is_read(self):
        raw = b"a,b,c,d,e,f\n1,2\n"  # malformed row follows; the header limit must fire first
        error = code_of(self, parse_table_csv, raw, TabularLimits(max_columns=5))
        self.assertEqual(error.code, tabular.TABLE_TOO_MANY_COLUMNS)

    def test_row_limit_stops_reading_as_soon_as_it_is_exceeded(self):
        raw = b"a,b\n1,2\n3,4\n5,6\n7,8\n9\n"  # the malformed last row is never reached
        error = code_of(self, parse_table_csv, raw, TabularLimits(max_rows=3))
        self.assertEqual(error.code, tabular.TABLE_TOO_MANY_ROWS)

    def test_row_and_column_limits_apply_to_in_memory_tables_before_scanning(self):
        rows = [{"a": 1, "y": "p"}] * 5 + [{"a": 1}]  # last row is malformed; the count check fires first
        self.assertEqual(code_of(self, profile_table, rows, target="y", limits=TabularLimits(max_rows=5)).code,
                         tabular.TABLE_TOO_MANY_ROWS)
        self.assertEqual(code_of(self, profile_table, {"a": [1, 2], "b": [1, 2], "y": ["p", "n"]}, target="y",
                                 limits=TabularLimits(max_columns=2)).code, tabular.TABLE_TOO_MANY_COLUMNS)

    def test_cardinality_limit_names_the_column_and_never_drops_it(self):
        rows = [{"city": f"c{i}", "x": float(i), "y": "pn"[i % 2]} for i in range(30)]
        error = code_of(self, profile_table, rows, target="y", limits=TabularLimits(max_categorical_cardinality=10))
        self.assertEqual(error.code, tabular.TABLE_CARDINALITY_EXCEEDED)
        self.assertEqual([item["column"] for item in error.details["violations"]], ["city"])
        self.assertIn("none were dropped", str(error))

    def test_high_cardinality_metadata_and_numeric_columns_do_not_violate(self):
        rows = [{"rowid": f"id{i}", "x": float(i), "y": "pn"[i % 2]} for i in range(60)]
        profile = profile_table(rows, target="y", id_columns=("rowid",), limits=TabularLimits(max_categorical_cardinality=10))
        self.assertEqual(profile["predictors"], ["x"])
        self.assertEqual(profile["columns"]["x"]["kind"], "numeric")

    def test_cardinality_counting_stops_at_the_cap(self):
        rows = [{"city": f"c{i}", "y": "pn"[i % 2]} for i in range(200)]
        error = code_of(self, profile_table, rows, target="y", limits=TabularLimits(max_categorical_cardinality=5))
        violation = error.details["violations"][0]
        self.assertIsNone(violation["distinct"])
        self.assertEqual(violation["distinct_at_least"], 6)

    def test_expansion_limit_uses_the_one_hot_width_estimate(self):
        rows = [{"a": f"a{i % 8}", "b": f"b{i % 8}", "c": f"c{i % 8}", "y": "pn"[i % 2]} for i in range(40)]
        profile_table(rows, target="y", limits=TabularLimits(max_expanded_features=24))
        error = code_of(self, profile_table, rows, target="y", limits=TabularLimits(max_expanded_features=23))
        self.assertEqual(error.code, tabular.TABLE_EXPANSION_EXCEEDED)
        self.assertEqual(error.details["violations"][0]["estimated_width"], 24)

    def test_error_records_are_json_safe(self):
        error = code_of(self, parse_table_csv, b"a,b\n1\n")
        json.dumps(error.to_record(), allow_nan=False)
        self.assertIsInstance(error, ValueError)


class ParsingDecisionsAndRejections(unittest.TestCase):
    def test_malformed_rows_are_rejected_not_dropped(self):
        for raw, reason in ((b"a,b\n1,2\n3\n4,5\n", "field_count_mismatch"),
                            (b"a,b\n1,2\n3,4,5\n", "field_count_mismatch"),
                            (b"a,b\n1,2\n\n3,4\n", "blank_line_inside_data"),
                            (b'a,b\n1,"2\n', "csv_syntax"),
                            (b'a,b\n1,"2"x\n', "csv_syntax")):
            with self.subTest(raw=raw):
                error = code_of(self, parse_table_csv, raw)
                self.assertEqual(error.code, tabular.TABLE_MALFORMED_ROW)
                self.assertEqual(error.details["reason"], reason)

    def test_field_count_error_reports_the_line_number(self):
        error = code_of(self, parse_table_csv, b"a,b\n1,2\n3\n4,5\n")
        self.assertEqual(error.details["line"], 3)
        self.assertIn("Line 3", str(error))

    def test_accepted_parse_records_zero_rejections_and_trailing_blank_lines(self):
        parsed = parse_table_csv(b"a,b\n1,2\n3,4\n\n\n", source_name="fixture.csv")
        self.assertEqual(parsed.row_count, 2)
        record = parsed.parsing
        self.assertEqual((record["rows_read"], record["rows_accepted"], record["rows_rejected"]), (2, 2, 0))
        self.assertEqual(record["trailing_blank_lines"], 2)
        self.assertEqual(record["row_exclusions"], [])
        self.assertEqual(record["malformed_row_policy"], "reject_with_error")
        self.assertEqual(len(record["input_sha256"]), 64)
        self.assertEqual(record["source_name"], "fixture.csv")
        self.assertEqual(parsed.columns, {"a": ["1", "3"], "b": ["2", "4"]})
        json.dumps(record, allow_nan=False)

    def test_delimiter_decisions_are_recorded(self):
        cases = ((b"a,b,c\n1,2,3\n4,5,6\n", ",", "consistent_field_count_scan"),
                 (b"a;b;c\n1;2;3\n4;5;6\n", ";", "consistent_field_count_scan"),
                 (b"a\tb\n1\t2\n", "\t", "consistent_field_count_scan"),
                 (b"a|b\n1|2\n", "|", "consistent_field_count_scan"),
                 (b"a;b\n1,5;2\n3,5;4\n", ";", "consistent_field_count_scan"),
                 (b'a,b\n"x,y",2\n', ",", "consistent_field_count_scan"),
                 (b"only\n1\n2\n", ",", "default_no_consistent_delimiter"))
        for raw, expected, method in cases:
            with self.subTest(raw=raw):
                decision = parse_table_csv(raw).parsing["delimiter"]
                self.assertEqual((decision["delimiter"], decision["method"]), (expected, method))
        explicit = parse_table_csv(b"a;b\n1;2\n", delimiter=";").parsing["delimiter"]
        self.assertEqual((explicit["delimiter"], explicit["method"]), (";", "user_specified"))
        self.assertEqual(parse_table_csv(b"a;b\n1,5;2\n3,5;4\n").columns["a"], ["1,5", "3,5"])

    def test_ambiguous_delimiter_is_flagged_and_resolved_by_fixed_order(self):
        decision = parse_table_csv(b"a,b;c,d\n1,2;3,4\n").parsing["delimiter"]
        self.assertTrue(decision["ambiguous"])
        self.assertEqual(decision["delimiter"], ",")

    def test_encoding_is_strict_utf8_with_bom_recorded_and_no_guessing(self):
        parsed = parse_table_csv(b"\xef\xbb\xbfa,b\n1,2\n")
        self.assertEqual(parsed.parsing["encoding"], {"encoding": "utf-8", "method": "strict_utf8", "bom_removed": True})
        self.assertEqual(list(parsed.columns), ["a", "b"])
        latin = "a,b\ncafé,2\n".encode("latin-1")
        self.assertEqual(code_of(self, parse_table_csv, latin).code, tabular.TABLE_ENCODING_UNDECODABLE)
        explicit = parse_table_csv(latin, encoding="latin-1")
        self.assertEqual(explicit.columns["a"], ["café"])
        self.assertEqual(explicit.parsing["encoding"]["method"], "user_specified")
        self.assertEqual(code_of(self, parse_table_csv, latin, encoding="no-such-codec").code, tabular.TABLE_ENCODING_UNKNOWN)

    def test_header_and_emptiness_rules(self):
        self.assertEqual(code_of(self, parse_table_csv, b"a,a\n1,2\n").code, tabular.TABLE_HEADER_INVALID)
        self.assertEqual(code_of(self, parse_table_csv, b"a,\n1,2\n").code, tabular.TABLE_HEADER_INVALID)
        self.assertEqual(code_of(self, parse_table_csv, b"").code, tabular.TABLE_EMPTY)
        self.assertEqual(code_of(self, parse_table_csv, b"  \n").code, tabular.TABLE_EMPTY)
        self.assertEqual(code_of(self, parse_table_csv, b"a,b\n").code, tabular.TABLE_NO_ROWS)
        self.assertEqual(code_of(self, parse_table_csv, "a,b\n1,2\n").code, tabular.TABLE_INPUT_INVALID)

    def test_missing_tokens_match_the_numeric_parser(self):
        parsed = parse_table_csv(b"x,y\n1,a\nNA,b\n?,a\nNaN,b\n,a\n")
        profile = profile_table(parsed, target="y")
        self.assertEqual(profile["columns"]["x"]["missing"], 4)
        self.assertEqual(profile["columns"]["x"]["kind"], "constant")

    def test_parsed_table_is_reusable_by_the_profile(self):
        parsed = parse_table_csv(b"a,y\n1,p\n2,n\n")
        self.assertIsInstance(parsed, ParsedTable)
        self.assertEqual(profile_table(parsed, target="y")["parsing"]["rows_read"], 2)


class OptionalDependencyBoundary(unittest.TestCase):
    def test_everything_but_table_to_dataframe_works_with_pandas_blocked(self):
        # scikit-learn itself may import pandas when it is installed, so block the import outright
        code = "\n".join([
            "import sys; sys.modules['pandas'] = None",
            "import rare_event_eval.core, rare_event_eval.policies, rare_event_eval.datasets",
            "from rare_event_eval.labels import LabelSpec, resolve_labels",
            "from rare_event_eval.tabular import parse_table_csv, profile_table",
            "from rare_event_eval.registry import REGISTRY",
            "t = parse_table_csv(b'a,y\\n1,p\\n2,n\\n')",
            "assert profile_table(t, target='y')['predictors'] == ['a']",
            "assert resolve_labels(['p', 'n', 'p', 'n'], LabelSpec('p')).y.tolist() == [1, 0, 1, 0]",
            "print('ok')"])
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual((result.returncode, result.stdout.strip()), (0, "ok"), result.stderr)

    def test_saved_prediction_core_sources_never_mention_pandas(self):
        import rare_event_eval.core as core
        import rare_event_eval.policies as policies
        for module in (core, policies):
            with open(module.__file__, encoding="utf-8") as stream:
                self.assertNotIn("pandas", stream.read(), module.__name__)


if __name__ == "__main__":
    unittest.main()
