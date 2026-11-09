"""The standalone HTML report of a ``training-1.1`` comparison: content, provenance, escaping and accessibility.

The tests render the recorded example (``examples/output/recorded-example.json``, a real app run saved by
``examples/make_recorded_example.py``) and variants of it. Nothing here trains a model.
"""
import copy
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import unittest

from rare_event_eval.reports import BIDI_CONTROLS, PROVENANCE_LABELS, render_html, training_report

RECORDED = Path(__file__).resolve().parents[1] / "examples" / "output" / "recorded-example.json"
HOSTILE = '<script>alert("x")</script>'
HOSTILE_IMG = '<img src=x onerror="alert(1)">'


def load() -> dict:
    return json.loads(RECORDED.read_text(encoding="utf-8"))


def replace_strings(value, mapping):
    """Copy of a JSON value with every string value that equals a key of ``mapping`` replaced (keys are untouched)."""
    if isinstance(value, dict):
        return {key: replace_strings(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_strings(item, mapping) for item in value]
    return mapping.get(value, value) if isinstance(value, str) else value


class Structure(HTMLParser):
    """Collects tags, the heading outline and table captions of a page."""

    def __init__(self):
        super().__init__()
        self.tags, self.headings, self.tables, self.captions, self.attributes = [], [], 0, 0, []
        self._heading = None

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attributes.extend((tag, name, value) for name, value in attrs)
        if tag == "table":
            self.tables += 1
        if tag == "caption":
            self.captions += 1
        if tag in ("h1", "h2", "h3"):
            self._heading = [tag, ""]

    def handle_data(self, data):
        if self._heading is not None:
            self._heading[1] += data

    def handle_endtag(self, tag):
        if self._heading is not None and tag == self._heading[0]:
            self.headings.append(tuple(self._heading))
            self._heading = None


class RecordedExampleReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = load()
        cls.report = training_report(cls.result)
        cls.page = render_html(cls.report)

    def test_the_recorded_example_is_a_labelled_app_run_with_no_historical_claim(self):
        self.assertEqual(self.result["provenance"], "recorded app run")
        self.assertEqual(self.result["schema_version"], "training-1.1")
        self.assertIn("recorded app run", self.page)
        self.assertIn("not a historical ERF research result", self.page)
        self.assertIn(self.result["app"]["app_version"], self.page)

    def test_every_required_fact_is_stated(self):
        result, page = self.result, self.page
        mapping, split = result["label_mapping"], result["split"]
        # label mapping and counts
        self.assertIn("Positive class: <strong>default</strong>", page)
        for item in mapping["original_labels"]:
            self.assertIn(f"<td>{item['label']}</td><td>{item['count']}</td>", page)
        counts = mapping["counts"]
        for text in (f"Rows in the file</th><td>{counts['input_rows']}</td>", f"Rows used (effective)</th><td>{counts['effective_rows']}</td>"):
            self.assertIn(text, page)
        # selected and excluded columns
        for name, kind in result["predictor_kinds"].items():
            self.assertIn(f"<td>{name}</td><td>{kind}</td>", page)
        self.assertIn("customer_id", page)
        self.assertIn("identifier", page.lower())
        # requested split, achieved partitions
        self.assertIn("Requested: <strong>stratified</strong> split, seed 7", page)
        for name, info in split["achieved"]["partitions"].items():
            self.assertIn(f"<td>{name}</td><td>{info['rows']}</td><td>{info['positive_rows']}</td><td>{info['negative_rows']}</td>", page)
        self.assertIn("same partitions for every model: yes", page)
        # models with recipe ids, capacity, status, time
        for requested, record in result["model_records"].items():
            self.assertIn(f"<th scope='row'>{requested}</th><td>{record['recipe_id']}</td>", page)
        self.assertIn("100 trees", page)
        self.assertIn("20 bags x 50 trees", page)
        self.assertIn("qualitative", page)
        self.assertIn("Fit seconds", page)
        # threshold source, metric definitions, review budget, costs, limitations
        self.assertIn("validation_only", page)
        self.assertIn("Test labels used", page)
        for term in ("Average precision", "ROC AUC", "Brier score, log loss", "Review budget policy"):
            self.assertIn(f"<dt>{term}</dt>", page)
        self.assertIn("Review budgets (rows flagged on the evaluated rows)", page)
        self.assertIn("assumptions, not measured savings", page)
        for limitation in self.report["limitations"]:
            self.assertIn(escape(limitation), page)

    def test_the_page_is_standalone(self):
        page = self.page
        self.assertNotRegex(page, r"https?://")
        self.assertNotIn("<script", page.lower())
        self.assertNotRegex(page, r"<link\b|<iframe\b|<object\b|<embed\b|@import|url\(")
        self.assertNotRegex(page, r"\b(?:src|href)\s*=\s*['\"](?!#)")
        self.assertIn("default-src 'none'", page)
        self.assertTrue(page.isascii())

    def test_headings_captions_language_and_landmarks(self):
        structure = Structure()
        structure.feed(self.page)
        self.assertIn('<html lang="en">', self.page)
        self.assertEqual(structure.tags.count("main"), 1)
        self.assertEqual([text for tag, text in structure.headings if tag == "h1"], ["Rare-event evaluation"])
        self.assertGreaterEqual(structure.tables, 8)
        self.assertEqual(structure.captions, structure.tables)   # every table has a caption
        h2 = [text for tag, text in structure.headings if tag == "h2"]
        for expected in ("Dataset and labels", "Columns", "Split", "Models, time and resources", "Threshold source",
                         "Model scores", "Decision policies", "Cost assumptions", "Metric definitions", "Checks and limitations"):
            self.assertIn(expected, h2)
        scrollers = [(name, value) for tag, name, value in structure.attributes if tag == "div" and name == "tabindex"]
        self.assertEqual(len(scrollers), structure.tables)   # a scrolling table can be reached with the keyboard
        self.assertGreater(self.page.count("scope='col'"), 0)
        self.assertIn('<a class="skip" href="#main">', self.page)

    def test_a_status_is_text_not_only_colour(self):
        self.assertIn("<td>success</td>", self.page)
        self.assertNotRegex(self.page, r"(?i)\b(?:green|red)\b")

    def test_rendering_is_deterministic(self):
        self.assertEqual(render_html(training_report(load())), self.page)


class FailuresAndEscaping(unittest.TestCase):
    def test_a_failed_model_is_listed_beside_the_successes(self):
        result = load()
        result["outcomes"]["erf"] = {"status": "timeout", "model": "erf", "timeout_seconds": 60, "reason_code": "MODEL_TIMEOUT",
                                     "model_record": result["model_records"]["erf"]}
        result["failures"] = [{"model": "erf", "status": "timeout", "reason_code": "MODEL_TIMEOUT",
                               "message": "Worker limit of 60 s reached."}]
        page = render_html(training_report(result))
        self.assertIn("<th scope='row'>erf</th>", page)
        self.assertIn("<td>timeout</td><td>MODEL_TIMEOUT</td>", page)
        self.assertIn("Worker limit of 60 s reached.", page)
        self.assertIn("<th scope='row'>logistic</th>", page)
        self.assertIn("<td>success</td>", page)
        self.assertIn("were not silently removed", page)
        self.assertIn("not replaced by another recipe", page)
        self.assertNotIn("Every requested model produced a result", page)

    def test_hostile_labels_columns_and_texts_are_escaped_everywhere(self):
        result = replace_strings(load(), {"default": HOSTILE, "repaid": HOSTILE_IMG, "customer_id": "<b>id</b>&co",
                                          "region": "re'gion", "Synthetic loan outcomes, 1,500 rows, five models": HOSTILE})
        result["failures"] = [{"model": "erf", "status": "failed", "reason_code": "MODEL_FIT_ERROR", "message": HOSTILE_IMG}]
        result["example"]["data_origin"] = HOSTILE
        page = render_html(training_report(result))
        self.assertNotIn("<script", page.lower())
        self.assertNotIn("<img", page.lower())
        self.assertNotIn("<b>id", page)
        self.assertIn("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;", page)
        self.assertIn("&lt;img src=x onerror=&quot;alert(1)&quot;&gt;", page)
        self.assertIn("&lt;b&gt;id&lt;/b&gt;&amp;co", page)
        self.assertIn("re&#x27;gion", page)
        self.assertNotRegex(page, r"<[a-z]+[^>]*\sonerror\s*=")

    def test_provenance_is_printed_only_when_it_is_one_of_the_four_labels(self):
        for label in PROVENANCE_LABELS:
            result = load()
            result["provenance"] = label
            self.assertIn(f"<strong>{label}</strong>", render_html(training_report(result)))
        for bad in (HOSTILE, "verified by Sherwan", "", None, 7):
            result = load()
            result["provenance"] = bad
            page = render_html(training_report(result))
            self.assertIn("<strong>not labelled</strong>", page)
            self.assertNotIn("<strong>verified by Sherwan</strong>", page)   # only the raw JSON dump may echo it, escaped
            self.assertNotIn("<script", page.lower())
        result = load()
        del result["provenance"]
        self.assertIn("not labelled", render_html(training_report(result)))

    def test_example_text_is_escaped_and_bounded(self):
        result = load()
        result["example"] = {"title": "<i>t</i>", "description": "x" * 5000, "unknown": "ignored", "note": HOSTILE}
        page = render_html(training_report(result))
        self.assertIn("&lt;i&gt;t&lt;/i&gt;", page)
        self.assertNotIn("ignored", page)
        self.assertNotIn("x" * 1300, page.replace("&quot;", ""))
        result["example"] = "not a mapping"
        self.assertNotIn("About this example", render_html(training_report(result)))

    def test_the_additive_fields_do_not_break_earlier_consumers(self):
        result = load()
        for key in ("provenance", "app", "example"):
            del result[key]
        report = training_report(result)
        self.assertNotIn("provenance", report)
        page = render_html(report)
        self.assertIn("not labelled", page)
        # a training-1.0 result has none of the 1.1 provenance fields and still renders
        old = copy.deepcopy(result)
        old["schema_version"] = "training-1.0"
        for key in ("label_mapping", "predictors", "excluded_columns", "input_parsing", "split", "model_records",
                    "preprocessing", "fallback_policy", "failures", "seed", "predictor_kinds"):
            old.pop(key, None)
        old_page = render_html(training_report(old))
        self.assertNotIn("Dataset and labels", old_page)
        self.assertIn("Model scores", old_page)
        with self.assertRaises(ValueError):
            bad = copy.deepcopy(result)
            bad["schema_version"] = "training-9.9"
            training_report(bad)


class DescriptionSource(unittest.TestCase):
    def test_a_recorded_run_names_its_real_source_not_a_user(self):
        page = render_html(training_report(load()))
        self.assertIn("Synthetic table generated by a seeded script in the repository.", page)
        self.assertIn("not typed by a user", page)
        self.assertNotIn("User-supplied", page)
        self.assertNotIn("User-declared description", page)

    def test_a_fresh_run_keeps_the_user_supplied_wording(self):
        result = load()
        result["provenance"] = "fresh app run"
        page = render_html(training_report(result))
        self.assertIn("User-supplied tabular training data", page)
        self.assertIn("(User-declared description.)", page)
        self.assertNotIn("seeded script", page)

    def test_an_unlabelled_report_keeps_the_user_declared_wording(self):
        result = load()
        del result["provenance"]
        page = render_html(training_report(result))
        self.assertIn("(User-declared description.)", page)

    def test_a_fresh_description_cannot_certify_its_own_source_with_keywords(self):
        result = load()
        result["provenance"] = "fresh app run"
        for row in result["outcomes"].values():
            row["report"]["data"]["description"] = "Synthetic benchmark table made by a seeded script of the project."
        page = render_html(training_report(result))
        self.assertIn("Synthetic benchmark table made by a seeded script of the project.", page)
        self.assertIn("(User-declared description.)", page)
        self.assertNotIn("not typed by a user", page)
        result = load()
        result["provenance"] = "fresh app run"
        for row in result["outcomes"].values():
            row["report"]["data"]["description"] = "My synthetic data"
        self.assertIn("(User-declared description.)", render_html(training_report(result)))


class BidiControls(unittest.TestCase):
    CODES = [0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069, 0x200E, 0x200F, 0x061C]

    def test_every_control_is_a_visible_marker_in_the_html_and_stays_real_in_the_json(self):
        chars = [chr(code) for code in self.CODES]
        self.assertEqual(sorted(BIDI_CONTROLS), sorted(self.CODES))
        label = "de" + "".join(chars) + "fault"
        result = replace_strings(load(), {"default": label, "customer_id": "id" + chars[4], "region": chars[5] + "region"})
        result["failures"] = [{"model": "erf", "status": "failed", "reason_code": "MODEL_FIT_ERROR", "message": "x" + chars[4] + "y"}]
        result["example"]["title"] = "T" + chars[4]
        page = render_html(training_report(result))
        for char, code in zip(chars, self.CODES):
            self.assertNotIn(char, page, hex(code))
            self.assertIn(f"&lt;U+{code:04X}&gt;", page, hex(code))
        self.assertIn("id&lt;U+202E&gt;", page)            # a column name
        self.assertIn("x&lt;U+202E&gt;y", page)            # a failure message
        self.assertIn("T&lt;U+202E&gt;", page)              # example text
        self.assertTrue(page.isascii() or not any(chr(code) in page for code in self.CODES))
        encoded = json.dumps(result, ensure_ascii=False)   # the JSON keeps the real characters
        for char in chars:
            self.assertIn(char, encoded)
        self.assertEqual(json.loads(encoded)["label_mapping"]["positive_label"], label)

    def test_saved_prediction_reports_show_the_markers_too(self):
        from rare_event_eval import evaluate_predictions
        label = "ev" + chr(0x202E) + "ent"
        report = evaluate_predictions([label, "clear"], {"m" + chr(0x2067): [.9, .1]}, positive_label=label,
                                      negative_label="clear", score_kind="probability")
        page = render_html(report)
        self.assertNotIn(chr(0x202E), page)
        self.assertNotIn(chr(0x2067), page)
        self.assertIn("ev&lt;U+202E&gt;ent", page)
        self.assertIn("m&lt;U+2067&gt;", page)

    def test_ordinary_text_is_untouched(self):
        from rare_event_eval.reports import escape as page_escape
        self.assertEqual(page_escape("plain"), "plain")
        self.assertEqual(page_escape("caf" + chr(0xE9) + chr(0x200B)), "caf" + chr(0xE9) + chr(0x200B))   # other characters stay
        self.assertEqual(page_escape("a<b&"), "a&lt;b&amp;")


class Wording(unittest.TestCase):
    def test_text_follows_the_writing_rules(self):
        page = render_html(training_report(load()))
        self.assertNotRegex(page, "[\u2013\u2014]")
        for banned in ("state-of-the-art", "best-in-class", "revolutionary", "guaranteed"):
            self.assertNotIn(banned, page.lower())
        self.assertNotRegex(page, r"[A-Za-z]:\\|/Users/|" + re.escape(Path.home().name))


if __name__ == "__main__":
    unittest.main()
