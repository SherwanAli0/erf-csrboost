"""Command-line evaluation of an explicitly labelled, UTF-8 CSV file."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import sys

from . import __version__
from .core import CostAssumptions, evaluate_predictions
from .reports import render_html


def parse_prediction_csv(raw: bytes, label_column: str, score_columns: list[str]) -> tuple[list, dict, str]:
    """Read exact labels and numeric scores; reject malformed rows rather than dropping them."""
    if not score_columns or len(score_columns) != len(set(score_columns)):
        raise ValueError("Supply at least one unique score column.")
    if label_column in score_columns:
        raise ValueError("The label column cannot also be a score column.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Input must be a UTF-8 CSV file.") from exc
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        headers = next(reader)
    except StopIteration as exc:
        raise ValueError("The input CSV is empty.") from exc
    if not headers or any(not header.strip() for header in headers) or len(headers) != len(set(headers)):
        raise ValueError("CSV headers must be nonempty and unique.")
    if any(column not in headers for column in [label_column, *score_columns]):
        raise ValueError("A requested label or score column is missing from the CSV header.")
    label_index = headers.index(label_column)
    score_indexes = {name: headers.index(name) for name in score_columns}
    labels, scores = [], {name: [] for name in score_columns}
    for row_number, row in enumerate(reader, start=2):
        if len(row) != len(headers):
            raise ValueError(f"CSV record {row_number} has the wrong field count; no rows were dropped.")
        labels.append(row[label_index])
        for name, index in score_indexes.items():
            try:
                scores[name].append(float(row[index]))
            except ValueError as exc:
                raise ValueError(f"CSV record {row_number} has a missing or nonnumeric score.") from exc
    return labels, scores, sha256(raw).hexdigest()


def load_csv(path: Path, label_column: str, score_columns: list[str]) -> tuple[list, dict, str]:
    return parse_prediction_csv(path.read_bytes(), label_column, score_columns)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rare-event-eval", description="Evaluate saved binary predictions locally. No training or threshold optimization.")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate", help="Evaluate one or more score columns in a UTF-8 CSV.")
    evaluate.add_argument("input", type=Path)
    evaluate.add_argument("--label-column", required=True)
    evaluate.add_argument("--score-column", action="append", required=True, help="Repeat for multiple models on the same rows.")
    evaluate.add_argument("--positive-label", required=True, help="Exact string label; never inferred from class frequency.")
    evaluate.add_argument("--negative-label", required=True)
    evaluate.add_argument("--score-kind", choices=("probability", "ranking"), required=True)
    evaluate.add_argument("--threshold", type=float, help="Fixed cutoff. Probability default: 0.5; ranking default: none.")
    evaluate.add_argument("--review-budget", type=int, action="append", default=[], help="Integer row count, including zero; repeat as needed.")
    evaluate.add_argument("--role", choices=("test", "validation", "unspecified"), default="unspecified")
    evaluate.add_argument("--data-description", help="Optional user-declared description, e.g. that inputs are fictional.")
    evaluate.add_argument("--false-positive-cost", type=float)
    evaluate.add_argument("--false-negative-cost", type=float)
    evaluate.add_argument("--review-cost", type=float, default=0.0)
    evaluate.add_argument("--calibration-bins", type=int, default=10)
    evaluate.add_argument("--output", type=Path, help="New JSON file; existing files are never overwritten. Default: stdout.")
    evaluate.add_argument("--html", type=Path, help="Optional new standalone HTML file.")
    from .training import MODEL_NAMES
    train = sub.add_parser("train", help="Run a bounded train/validation/test comparison locally. Numeric predictors use the "
                                         "numeric recipes; categorical predictors use the mixed-type adapter (needs pandas, extra [app]).")
    train.add_argument("input", type=Path)
    train.add_argument("--target", required=True)
    train.add_argument("--positive-label", required=True)
    train.add_argument("--negative-label", help="Required unless --one-vs-rest is given.")
    train.add_argument("--one-vs-rest", action="store_true",
                       help="Treat every label other than --positive-label as negative; the original-to-binary mapping is recorded.")
    train.add_argument("--missing-target", choices=("error", "drop_reported"), default="error",
                       help="error (default) rejects missing targets; drop_reported excludes those rows and lists them.")
    train.add_argument("--predictor", action="append", help="Use only these predictor columns (repeat). Others are listed as excluded.")
    train.add_argument("--exclude-column", action="append", help="Exclude a column from the predictors (repeat).")
    train.add_argument("--id-column", action="append", help="Identifier column, never a predictor (repeat).")
    train.add_argument("--categorical", choices=("auto", "reject"), default="auto",
                       help="auto (default): use the mixed-type adapter when a predictor is not numeric; reject: refuse such files.")
    train.add_argument("--max-categories", type=int, help="Largest number of distinct values per categorical predictor (default 50).")
    train.add_argument("--max-expanded-features", type=int, help="Largest one-hot expanded predictor width (default 400).")
    train.add_argument("--model", choices=MODEL_NAMES, action="append")
    train.add_argument("--split", choices=("stratified", "group", "feature_group", "chronological"), default="stratified")
    train.add_argument("--group-column")
    train.add_argument("--time-column")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--calibration", choices=("none", "sigmoid"), default="none")
    train.add_argument("--data-description", default="User-supplied numeric training data; provenance is not independently verified.")
    train.add_argument("--review-fraction", type=float, action="append")
    train.add_argument("--timeout", type=float, default=60)
    train.add_argument("--false-positive-cost", type=float)
    train.add_argument("--false-negative-cost", type=float)
    train.add_argument("--review-cost", type=float, default=0.)
    train.add_argument("--output", type=Path)
    train.add_argument("--html", type=Path)
    serve = sub.add_parser("serve", help="Open the full local workbench on the loopback interface.")
    serve.add_argument("--port", type=int, default=8601)
    export = sub.add_parser("export-demo", help="Copy the free static browser demo into a new directory.")
    export.add_argument("--output-directory", type=Path, required=True)
    return parser


def _write_new_files(payloads: list[tuple[Path, str]], input_path: Path) -> None:
    resolved = [path.resolve() for path, _ in payloads]
    if len(set(resolved)) != len(resolved) or input_path.resolve() in resolved:
        raise ValueError("Output paths must differ from each other and from the input file.")
    if any(path.exists() for path in resolved):
        raise ValueError("An output path already exists. Choose new filenames; nothing was overwritten.")
    created = []
    try:
        for path, content in payloads:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                created.append(path)
                stream.write(content)
    except OSError:
        for path in created:
            path.unlink(missing_ok=True)
        raise


NUMERIC_DESCRIPTION = "User-supplied numeric training data; provenance is not independently verified."


def _train(args, costs) -> int:
    from .datasets import NonNumericFeatureError, parse_training_csv
    from .training import DEFAULT_MODELS, train_comparison
    from .reports import training_report
    from .tabular import excluded_metadata_records
    if not args.data_description.strip() or len(args.data_description) > 500:
        raise ValueError('Data description must contain 1..500 characters.')
    if args.one_vs_rest == (args.negative_label is not None):
        raise ValueError("Give --negative-label for a binary target, or --one-vs-rest with no --negative-label.")
    raw = args.input.read_bytes()
    use_adapter = any((args.predictor, args.exclude_column, args.id_column, args.one_vs_rest,
                       args.missing_target != "error", args.max_categories is not None,
                       args.max_expanded_features is not None))
    if not use_adapter:
        try:
            dataset = parse_training_csv(raw, target=args.target, group_column=args.group_column, time_column=args.time_column)
        except NonNumericFeatureError:
            if args.categorical == "reject":
                raise
            use_adapter = True
    if use_adapter:
        from .labels import LabelSpec
        from .mixed import train_comparison_tabular
        from .tabular import TabularLimits
        defaults = TabularLimits()
        limits = TabularLimits(max_categorical_cardinality=args.max_categories or defaults.max_categorical_cardinality,
                               max_expanded_features=args.max_expanded_features or defaults.max_expanded_features)
        spec = (LabelSpec(args.positive_label, mode="one_vs_rest") if args.one_vs_rest
                else LabelSpec(args.positive_label, args.negative_label))
        result = train_comparison_tabular(
            raw, target=args.target, label_spec=spec, predictors=args.predictor, exclude_columns=args.exclude_column or (),
            group_column=args.group_column, time_column=args.time_column, id_columns=tuple(args.id_column or ()),
            split_mode=args.split, seed=args.seed, models=args.model or DEFAULT_MODELS, calibration=args.calibration,
            limits=limits, timeout_seconds=args.timeout, missing_policy=args.missing_target,
            review_fractions=args.review_fraction or (.1, .2), costs=costs,
            description=None if args.data_description == NUMERIC_DESCRIPTION else args.data_description,
            categorical_handling=args.categorical, require_pandas_for_categorical=True)
    else:
        result = train_comparison(dataset['X'], dataset['labels'], models=args.model or DEFAULT_MODELS,
            positive_label=args.positive_label, negative_label=args.negative_label, split_mode=args.split,
            groups=dataset['groups'], times=dataset['times'], seed=args.seed, calibration=args.calibration,
            review_fractions=args.review_fraction or (.1,.2), costs=costs, timeout_seconds=args.timeout,
            report_context={'feature_names': dataset['feature_names'], 'input_sha256': dataset['input_sha256'],
                            'parser': 'parse_training_csv (numeric, strict; malformed rows raise)',
                            'excluded_columns': excluded_metadata_records(args.target, args.group_column, args.time_column)})
        result['input_sha256'] = dataset['input_sha256']
        result['feature_names'] = dataset['feature_names']
        for row in result['outcomes'].values():
            if row['status']=='success':
                row['report']['data']['description']=args.data_description+' Fresh fit with separate evaluation partitions.'
                row['execution'].get('model_parameters',{}).pop('cv',None)
    output = json.dumps(result, indent=2, allow_nan=False) + '\n'
    payloads = [(args.output, output)] if args.output else []
    if args.html:
        payloads.append((args.html, render_html(training_report(result))))
    _write_new_files(payloads, args.input)
    if args.output is None:
        sys.stdout.write(output)
    else:
        print('Saved the local training comparison with explicit model outcomes.', file=sys.stderr)
    return 0 if all(row['status']=='success' for row in result['outcomes'].values()) else 1


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            from .server import serve
            serve(args.port)
            return 0
        if args.command == "export-demo":
            from .server import export_demo
            export_demo(args.output_directory)
            print("Static demo exported. Nothing was published or deployed.")
            return 0
        cost_flags = (args.false_positive_cost, args.false_negative_cost)
        if any(value is not None for value in cost_flags) and not all(value is not None for value in cost_flags):
            raise ValueError("Supply both --false-positive-cost and --false-negative-cost, or neither.")
        if not any(value is not None for value in cost_flags) and args.review_cost != 0:
            raise ValueError("--review-cost also requires both error-cost assumptions.")
        costs = (CostAssumptions(*cost_flags, args.review_cost)
                 if all(value is not None for value in cost_flags) else None)
        if args.command == "train":
            return _train(args, costs)
        labels, scores, input_hash = load_csv(args.input, args.label_column, args.score_column)
        report = evaluate_predictions(
            labels, scores, positive_label=args.positive_label, negative_label=args.negative_label,
            score_kind=args.score_kind, threshold=args.threshold, review_budgets=args.review_budget,
            evaluation_role=args.role, costs=costs, calibration_bins=args.calibration_bins,
            data_description=args.data_description,
        )
        report["data"]["input_sha256"] = input_hash
        output = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        payloads = []
        if args.output:
            payloads.append((args.output, output))
        if args.html:
            payloads.append((args.html, render_html(report)))
        _write_new_files(payloads, args.input)
        if args.output is None:
            sys.stdout.write(output)
        else:
            print(f"Evaluated {len(labels)} rows and {len(scores)} model(s). Saved new report file(s).", file=sys.stderr)
        return 0
    except (ValueError, OSError, csv.Error, OverflowError) as exc:
        print(f"rare-event-eval: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
