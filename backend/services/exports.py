"""Exports of a finished training job: provenance label, effective settings and the reproducibility bundle.

Nothing here computes a metric or a split. The functions take the job's ``training-1.1`` report and the validated
job request and produce files:

``annotate_report``     add the provenance label and the app block to a report (additive fields only)
``effective_settings``  the request with every default written out (the values the run really used)
``replay_argv``         the exact ``python -m rare_event_eval train`` arguments that repeat the run
``build_repro_bundle``  a zip (config, report JSON and HTML, versions, README, replay command), built in memory

The bundle holds aggregates and settings only. It never holds the uploaded table, a row, a prediction, a split
index list or a model. It is a pure function of its inputs: the same report and settings give the same bytes (fixed
file order, fixed timestamps, fixed compression level) for one Python and zlib build.
"""
from __future__ import annotations

import importlib.metadata
import inspect
import io
import json
import re
import shlex
import zipfile

from .errors import ServiceError

PROVENANCE_LABELS = ("fictional fixture", "recorded app run", "fresh app run", "historical research record")
FRESH_RUN = "fresh app run"
RECORDED_RUN = "recorded app run"

REPORT_SCHEMA = "training-1.1"
CONFIG_SCHEMA = "fyd-job-config-1.0"
VERSIONS_SCHEMA = "fyd-versions-1.0"
BUNDLE_SCHEMA = "fyd-repro-bundle-1.0"
BUNDLE_FILES = ("config.json", "report.json", "report.html", "versions.json", "README.txt", "replay.txt")
MAX_BUNDLE_UNCOMPRESSED_BYTES = 40 * 1024 * 1024
MAX_BUNDLE_BYTES = 20 * 1024 * 1024
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ZIP_LEVEL = 6

INPUT_PLACEHOLDER = "YOUR_INPUT_FILE.csv"
OUTPUT_JSON = "replay-report.json"
OUTPUT_HTML = "replay-report.html"
CLI_SPLIT_MODES = ("stratified", "group", "feature_group", "chronological")
CLI_CALIBRATIONS = ("none", "sigmoid")
CLI_DEFAULT_MAX_BYTES = 5_000_000   # TabularLimits() default; the command line has no flag for it
CLI_DEFAULT_MAX_ROWS = 20_000       # the same, for rows
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_POWERSHELL_PLAIN = re.compile(r"^[A-Za-z0-9_./:=+-]+$")


class ExportError(ServiceError):
    """An export could not be produced (for example it would be larger than the bound)."""


def escape_markup_in_json(data: bytes) -> bytes:
    """Write ``<``, ``>`` and ``&`` as JSON escapes. They occur only inside strings, so the JSON is unchanged."""
    return data.replace(b"<", b"\\u003c").replace(b">", b"\\u003e").replace(b"&", b"\\u0026")


def dump_json(value) -> bytes:
    """Strict, ASCII-only JSON with a final newline and no raw markup characters (same as the HTTP exports)."""
    return escape_markup_in_json((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("ascii"))


# ------------------------------------------------------------------------------------------ report fields

def annotate_report(report: dict, provenance: str = FRESH_RUN, app: dict | None = None) -> dict:
    """Copy of ``report`` with ``provenance`` (one of the four labels) and, when given, an ``app`` block.

    The fields are additive: every ``training-1.1`` field keeps its meaning, so a consumer that does not know
    them ignores them. ``app`` carries the application version and the source release id of the run.
    """
    if provenance not in PROVENANCE_LABELS:
        raise ValueError(f"provenance must be one of {PROVENANCE_LABELS}")
    out = dict(report)
    out["provenance"] = provenance
    if app:
        out["app"] = {"app_version": str(app.get("app_version") or "unknown"),
                      "release_id": str(app.get("release_id") or "unknown"),
                      "note": "Application that produced this report. The report itself comes from the "
                              "rare-event-eval package."}
    return out


def _train_defaults() -> dict:
    from .training import app_default_models, train_for_app
    values = {name: item.default for name, item in inspect.signature(train_for_app).parameters.items()
              if item.default is not inspect.Parameter.empty}
    values["models"] = list(app_default_models())
    return values


def effective_settings(params: dict, limits: dict, model_timeout_seconds: float) -> dict:
    """The job request with every default filled in (the defaults are read from ``train_for_app`` itself)."""
    defaults = _train_defaults()
    params = dict(params or {})

    def pick(name):
        value = params.get(name)
        return defaults.get(name) if value is None else value

    costs = params.get("costs")
    one_vs_rest = bool(params.get("one_vs_rest"))
    return {
        "target": params.get("target"), "positive_label": params.get("positive_label"),
        "negative_label": None if one_vs_rest else params.get("negative_label"), "one_vs_rest": one_vs_rest,
        "models": list(pick("models")), "split_mode": pick("split_mode"), "seed": pick("seed"),
        "calibration": pick("calibration"), "missing_policy": pick("missing_policy"),
        "review_fractions": [float(item) for item in pick("review_fractions")],
        "costs": dict(costs) if isinstance(costs, dict) else None,
        "id_columns": list(params.get("id_columns") or []), "exclude_columns": list(params.get("exclude_columns") or []),
        "group_column": params.get("group_column"), "time_column": params.get("time_column"),
        "per_model_timeout_seconds": float(model_timeout_seconds),
        "table_limits": {"max_bytes": limits.get("max_bytes"), "max_rows": limits.get("max_rows"),
                         "max_expanded_features": limits.get("max_expanded_features")},
    }


# ------------------------------------------------------------------------------------------ replay command

def _text(value) -> str:
    """A label or setting as the command line carries it (the package compares labels as text)."""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def replay_argv(settings: dict, input_name: str = INPUT_PLACEHOLDER) -> tuple:
    """``(arguments, limits_the_command_cannot_express)`` for ``python -m rare_event_eval``.

    Every value option is written as ``--name=value`` so a label or column name that begins with a dash cannot be
    read as another option. The command always carries explicit values (seed, split, calibration, review fractions,
    timeout, missing-target policy) instead of relying on defaults, and ``--max-expanded-features`` is always given
    because that option makes the command line use the mixed-type adapter, the code path the hosted job runs.
    """
    args = ["python", "-m", "rare_event_eval", "train", input_name, f"--target={settings['target']}",
            f"--positive-label={_text(settings['positive_label'])}"]
    if settings["one_vs_rest"]:
        args.append("--one-vs-rest")
    else:
        args.append(f"--negative-label={_text(settings['negative_label'])}")
    args.append(f"--missing-target={settings['missing_policy']}")
    args += [f"--model={name}" for name in settings["models"]]
    args.append(f"--split={settings['split_mode']}")
    if settings["group_column"]:
        args.append(f"--group-column={settings['group_column']}")
    if settings["time_column"]:
        args.append(f"--time-column={settings['time_column']}")
    args += [f"--seed={settings['seed']}", f"--calibration={settings['calibration']}"]
    args += [f"--review-fraction={_text(item)}" for item in settings["review_fractions"]]
    args.append(f"--timeout={_text(settings['per_model_timeout_seconds'])}")
    args += [f"--id-column={name}" for name in settings["id_columns"]]
    args += [f"--exclude-column={name}" for name in settings["exclude_columns"]]
    costs = settings["costs"]
    if costs:
        args += [f"--false-positive-cost={_text(costs['false_positive_cost'])}",
                 f"--false-negative-cost={_text(costs['false_negative_cost'])}"]
        if costs.get("review_cost") is not None:
            args.append(f"--review-cost={_text(costs['review_cost'])}")
    args.append(f"--max-expanded-features={settings['table_limits']['max_expanded_features']}")
    args += [f"--output={OUTPUT_JSON}", f"--html={OUTPUT_HTML}"]
    unsupported = []
    if settings["split_mode"] not in CLI_SPLIT_MODES:
        unsupported.append(f"split mode {settings['split_mode']!r} is not a command line choice")
    if settings["calibration"] not in CLI_CALIBRATIONS:
        unsupported.append(f"calibration {settings['calibration']!r} is not a command line choice")
    return args, unsupported


def posix_quote(args) -> str:
    return " ".join(shlex.quote(item) for item in args)


def powershell_quote(args) -> str | None:
    """Single-quoted PowerShell words (a single quote is doubled); plain words stay unquoted.

    Returns ``None`` when a value holds a double quote: Windows PowerShell 5.1 drops or mangles an embedded double
    quote when it starts a program, so no quoting can be promised there. The JSON argument list stays exact."""
    if any('"' in item for item in args):
        return None
    words = []
    for index, item in enumerate(args):
        if index == 0 or _POWERSHELL_PLAIN.match(item):
            words.append(item)
        else:
            words.append("'" + item.replace("'", "''") + "'")
    return " ".join(words)


def _has_control(args) -> bool:
    return any(_CONTROL.search(item) for item in args)


# ------------------------------------------------------------------------------------------ versions

def _library(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def versions_record(report: dict, app: dict | None) -> dict:
    """App, package, schema, interpreter and library versions plus recipe ids. Read-only facts, no paths."""
    app = app or {}
    tool = {}
    for row in (report.get("outcomes") or {}).values():
        if isinstance(row, dict) and row.get("status") == "success":
            tool = dict(row.get("report", {}).get("tool") or {})
            break
    recipes = {}
    for requested, record in (report.get("model_records") or {}).items():
        if isinstance(record, dict):
            recipes[str(requested)] = {"recipe_id": record.get("recipe_id"), "recipe_version": record.get("version"),
                                       "scope": record.get("scope"), "family": record.get("family")}
    return {
        "schema": VERSIONS_SCHEMA,
        "app_version": str(app.get("app_version") or "unknown"),
        "source_release_id": str(app.get("release_id") or "unknown"),
        "package": {"name": tool.get("name", "rare-event-eval"), "version": tool.get("version")},
        "report_schema": report.get("schema_version"), "job_config_schema": CONFIG_SCHEMA,
        "python": tool.get("python"),
        "libraries": {"numpy": tool.get("numpy"), "scikit_learn": tool.get("scikit_learn"),
                      "imbalanced_learn": _library("imbalanced-learn"), "pandas": _library("pandas")},
        "recipes": recipes,
        "note": "numpy, scikit-learn and Python are the versions the models ran under (recorded in the report). "
                "imbalanced-learn and pandas were read in the web process of the same installation.",
    }


# ------------------------------------------------------------------------------------------ text files

def _line(value, limit: int = 240) -> str:
    text = _CONTROL.sub(" ", str(value))
    return text if len(text) <= limit else text[: limit - 3] + "..."


def replay_notes(report: dict, settings: dict, unsupported: list) -> dict:
    """Facts the reader needs about this particular run: what the command repeats and where it cannot."""
    parsing = ((report.get("input_parsing") or {}).get("details") or {})
    cannot = list(unsupported)
    caution = []
    notes = [item for item in parsing.get("format_notes") or [] if item]
    warnings = [item for item in parsing.get("read_warnings") or [] if item]
    if notes:
        caution.append("The app converted the uploaded file before reading it (" + _line("; ".join(map(str, notes))) +
                       "). The command line reads UTF-8 delimited text only, so save the table as UTF-8 CSV first. "
                       "The input hash and the parsing record will then differ from this run.")
    if warnings:
        caution.append("The app's table reader reported: " + _line("; ".join(map(str, warnings))) +
                       ". The command line reader rejects a malformed row instead of skipping it, so it may stop "
                       "or read a different table.")
    size = parsing.get("input_bytes")
    if isinstance(size, int) and size > CLI_DEFAULT_MAX_BYTES:
        cannot.append(f"the input is {size} bytes; the command line has no option for the byte limit and refuses "
                      f"files above {CLI_DEFAULT_MAX_BYTES} bytes")
    if (report.get("failures") or []):
        caution.append("Some models failed or timed out in this run. A timeout depends on the speed of the machine, "
                       "so a replay can finish a model that timed out here, or the other way round.")
    limits = settings["table_limits"]
    cannot.append(f"the row limit of the app ({limits.get('max_rows')} rows) and its byte limit "
                  f"({limits.get('max_bytes')} bytes): the command line uses the package defaults "
                  f"({CLI_DEFAULT_MAX_ROWS} rows, {CLI_DEFAULT_MAX_BYTES} bytes), which only matters for a "
                  "larger file")
    return {"cannot": cannot, "caution": caution}


def replay_text(report: dict, settings: dict, versions: dict) -> str:
    args, unsupported = replay_argv(settings)
    notes = replay_notes(report, settings, unsupported)
    digest = report.get("input_sha256") or "not recorded"
    parsing = ((report.get("input_parsing") or {}).get("details") or {})
    lines = [
        "Replay of this analysis with the rare-event-eval package",
        "========================================================",
        "",
        "This repeats the model comparison of the report in this bundle on the same data, with the same settings.",
        "The bundle does not contain the data. You need the file you uploaded.",
        "",
        "1. Install the package (Python 3.11 or newer). From a copy of the source repository:",
        "",
        "       pip install \"./evaluation[app,training]\"",
        "",
        f"   Source release of the run that made this bundle: {versions['source_release_id']}",
        f"   Package version of the run: {versions['package']['version']}",
        "",
        "2. Put your original file next to this text. It must be the exact file that was uploaded:",
        f"   SHA-256 {digest}, {parsing.get('input_bytes', 'unknown')} bytes.",
        f"   Name it {INPUT_PLACEHOLDER} or change the name in the command. Check the hash first, for example:",
        f"       python -c \"import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())\" "
        f"{INPUT_PLACEHOLDER}",
        "",
        "3. Run the command. It writes two new files and never overwrites an existing file.",
        ""]
    if _has_control(args):
        lines += ["   A value in this run contains a line break or another control character, which cannot be "
                  "written safely on one command line. Use the argument list below with a script.", ""]
    else:
        shell = powershell_quote(args)
        lines += ["   POSIX shell (bash, zsh):", "", "       " + posix_quote(args), "", "   Windows PowerShell:", ""]
        lines += (["       " + shell, ""] if shell is not None else
                  ["       Not shown: a value in this run contains a double quote, which Windows PowerShell 5.1 "
                   "cannot pass safely.", "       Use the POSIX command in a POSIX shell, or the argument list below.", ""])
    lines += ["   The same arguments as a JSON list (exact; use it to start the program from a script):", "",
              "       " + json.dumps(args), ""]
    lines += ["What the command repeats",
              "------------------------",
              "Models, split mode and seed, calibration, positive and negative label (or one versus rest), target,",
              "excluded and identifier columns, group and time columns, review fractions, error and review costs,",
              "the missing-target policy, the per-model time limit and the width limit for expanded predictors.",
              "The command uses the same mixed-type code path as the hosted job.",
              "",
              "What should match and what may differ",
              "-------------------------------------",
              "The metrics, thresholds, class counts, split counts and index hashes, preprocessing records and model",
              "records match when the input is read the same way. These fields differ by design: creation times,",
              "fit and wall-clock seconds, peak memory, the parsing record (the app reads uploads with its own table",
              "reader; the command line uses the package reader) and the table limits that are echoed back. The",
              "provenance label and app block of this bundle's report are added by the app, not by the package.",
              "",
              "What the command cannot express"]
    lines.append("-" * len("What the command cannot express"))
    lines += ["- " + _line(item, 400) for item in notes["cannot"]]
    if notes["caution"]:
        lines += ["", "Cautions for this run", "---------------------"] + ["- " + _line(item, 600) for item in notes["caution"]]
    lines += ["", "Exit status: 0 when every model finished, 1 when at least one model failed (the files are still "
                  "written), 2 for an input or option error.", ""]
    return "\n".join(lines)


def readme_text(report: dict, versions: dict) -> str:
    provenance = report.get("provenance") or "not stated"
    return "\n".join([
        "FixYourDataset reproducibility bundle",
        "=====================================",
        "",
        f"Provenance of the result: {provenance}.",
        f"Bundle format: {BUNDLE_SCHEMA}. Report schema: {report.get('schema_version')}.",
        f"App version: {versions['app_version']}. Source release: {versions['source_release_id']}.",
        "",
        "Files",
        "-----",
        "config.json    The settings of the run (the request, every default written out, the limits).",
        "report.json    The full report of the run: aggregate metrics, label mapping, split counts, models.",
        "report.html    The same report as a standalone page. It has no script and loads nothing from a network.",
        "versions.json  App, package, schema, Python and library versions and the recipe id of every model.",
        "replay.txt     The exact command that repeats the run with the rare-event-eval package.",
        "README.txt     This file.",
        "",
        "What is not in this bundle",
        "--------------------------",
        "The uploaded table, any row of it, any prediction or score per row, the row indexes of the split (only",
        "their hashes and counts are in the report) and any fitted model. Nothing here can be turned back into",
        "your data. Aggregates and column or label names can still be sensitive, for example when a label or a",
        "column name is itself a personal detail, so share the bundle deliberately.",
        "",
        "What it is for",
        "--------------",
        "It lets you check that the report matches a run you can repeat on your own machine, and lets someone else",
        "see exactly what was run. It does not show that the result will hold on other data, and these models",
        "were compared on one held-out partition of one table. See the limitations in report.html.",
        "A result produced by this app is not a historical research result of the ERF study.",
        ""])


# ------------------------------------------------------------------------------------------ the bundle

def bundle_members(report: dict, settings: dict, config_body: dict, report_html: str, app: dict | None) -> dict:
    """The six files of the bundle as ``{name: bytes}`` in bundle order."""
    versions = versions_record(report, app)
    config = dict(config_body)
    config["bundle_schema"] = BUNDLE_SCHEMA
    files = {
        "config.json": dump_json(config),
        "report.json": dump_json(report),
        "report.html": report_html.encode("utf-8"),
        "versions.json": dump_json(versions),
        "README.txt": readme_text(report, versions).encode("utf-8"),
        "replay.txt": replay_text(report, settings, versions).encode("utf-8"),
    }
    assert tuple(files) == BUNDLE_FILES
    return files


def build_repro_bundle(report: dict, settings: dict, config_body: dict, report_html: str, app: dict | None) -> bytes:
    """Zip of ``bundle_members``: deterministic order, timestamps and attributes; bounded in size."""
    files = bundle_members(report, settings, config_body, report_html, app)
    if sum(len(item) for item in files.values()) > MAX_BUNDLE_UNCOMPRESSED_BYTES:
        raise ExportError("EXPORT_TOO_LARGE", "The bundle would be larger than the service allows.")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=ZIP_LEVEL)
    data = buffer.getvalue()
    if len(data) > MAX_BUNDLE_BYTES:
        raise ExportError("EXPORT_TOO_LARGE", "The bundle would be larger than the service allows.")
    return data

