"""Strict numeric training CSV parsing. No inferred targets or automatic encoding."""
import csv
from datetime import datetime, timezone
from hashlib import sha256
import io
import numpy as np


class NonNumericFeatureError(ValueError):
    """A predictor cell is not numeric. The CLI uses this to switch to the mixed-type adapter."""


def parse_training_csv(raw, *, target, group_column=None, time_column=None):
    text = raw.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ValueError("Training CSV is empty.") from exc
    if len(set(header)) != len(header) or any(not name.strip() for name in header):
        raise ValueError("Training CSV headers must be nonempty and unique.")
    declared = [name for name in (target, group_column, time_column) if name is not None]
    if len(set(declared)) != len(declared) or any(name not in header for name in declared):
        raise ValueError("Target, group and time columns must be distinct existing columns.")
    features = [name for name in header if name not in declared]
    if not features:
        raise ValueError("At least one explicit numeric feature is required.")
    X, labels, groups, times = [], [], [], []
    for index, row in enumerate(reader, start=2):
        if len(row) != len(header):
            raise ValueError(f"Training CSV record {index} has the wrong field count.")
        record = dict(zip(header, row))
        if not record[target].strip():
            raise ValueError(f"Training CSV record {index} has a missing label.")
        try:
            values = [float(record[name]) if record[name].strip() not in ("", "?", "NA", "NaN") else np.nan for name in features]
        except ValueError as exc:
            raise NonNumericFeatureError(f"Training CSV record {index} contains a nonnumeric feature. Automatic categorical encoding is not implemented.") from exc
        X.append(values)
        labels.append(record[target])
        if group_column:
            if not record[group_column].strip():
                raise ValueError("Group identifiers cannot be empty.")
            groups.append(record[group_column])
        if time_column:
            value = record[time_column]
            try:
                numeric_time = float(value)
            except ValueError:
                try:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    numeric_time = parsed.replace(tzinfo=timezone.utc).timestamp() if parsed.tzinfo is None else parsed.timestamp()
                except ValueError as exc:
                    raise ValueError("Time values must be numeric or ISO dates/timestamps; naive dates are interpreted as UTC.") from exc
            times.append(numeric_time)
    array = np.asarray(X, dtype=float)
    if array.ndim != 2 or np.isinf(array).any():
        raise ValueError("Training CSV must contain numeric features without infinity.")
    return {"X": array, "labels": np.asarray(labels), "feature_names": features,
            "groups": np.asarray(groups) if group_column else None,
            "times": np.asarray(times) if time_column else None,
            "input_sha256": sha256(raw).hexdigest()}
