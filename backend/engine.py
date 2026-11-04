"""
engine.py - leakage-safe class-imbalance fixer engine.

Built on scikit-learn and imbalanced-learn. It includes the plain baseline, a port of the published
CSRBoost method, and ERF-CSRBoost, an experimental, unpublished extension of CSRBoost (a bagged
ensemble of resampled random forests, not a boosting method). The ERF here is adapted from the
research implementation and is not numerically identical to it. The whole point of this tool is
CORRECTNESS: split first, resample the TRAIN split only, never touch the test set.
"""
import io
import math
import warnings
import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score, accuracy_score,
                             balanced_accuracy_score)

from imblearn.over_sampling import (RandomOverSampler, SMOTE, SMOTENC,
                                    BorderlineSMOTE, ADASYN)
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from imblearn.combine import SMOTETomek
# NOTE: imblearn.metrics.geometric_mean_score triggers a sklearn/imblearn _check_targets
# version clash on some builds; we compute G-Mean directly from per-class recall instead.
def geometric_mean_score(y_true, y_pred, average=None):
    rec = np.asarray(recall_score(y_true, y_pred, average=None, zero_division=0), dtype=float)
    return float(np.prod(rec) ** (1.0 / len(rec))) if len(rec) else 0.0

SEED = 42
TEST_SIZE = 0.25
N_BAGS = 20            # ERF-CSRBoost bags (same count as the research setting)
N_TREES = 50
ID_FRAC = 0.5          # object col with unique/n above this is treated as ID / free text
NUMERIC_ONLY = {"borderline_smote", "adasyn", "smote_tomek", "cluster_centroids"}

METHODS = {
    "clustered_smote":   "Clustered undersampling + SMOTE (CSRBoost method)",
    "none":              "No resampling (use class weights instead)",
    "random_over":       "Random oversampling (copy minority rows; any data type)",
    "random_under":      "Random undersampling (drop majority rows; any data type)",
    "smote":             "SMOTE (synthetic minority; auto SMOTE-NC if categoricals)",
    "borderline_smote":  "Borderline-SMOTE (numeric only)",
    "adasyn":            "ADASYN (numeric only)",
    "smote_tomek":       "SMOTE + Tomek cleaning (numeric only)",
    "cluster_centroids": "ClusterCentroids undersampling (numeric only)",
}
CLASSIFIERS = {
    "logreg":        "Logistic Regression",
    "random_forest": "Random Forest",
    "adaboost":      "AdaBoost",
}


TARGET_HINTS = {"target", "class", "label", "outcome", "result", "y", "churn", "churned",
                "fraud", "is_fraud", "default", "defaulted", "diagnosis", "status", "category",
                "type", "species", "survived", "spam", "condition", "attrition", "approved",
                "malignant", "benign", "disease", "risk", "quality"}
# names that signal a value to PREDICT AS A NUMBER (regression) -> not a classification label
REGRESSION_HINTS = {"price", "amount", "cost", "value", "revenue", "sales", "salary", "income",
                    "temperature", "temp", "score", "rating", "total", "charge", "fare", "premium",
                    "msrp", "mpg", "sale_price", "saleprice", "target_value"}
NUM_CLASS_MAX = 15          # a numeric target with more distinct values looks like a score/continuous
MAX_CLASSES = 20            # more distinct classes than this looks like an ID / free text / multi-label


def _promote_double_header(df):
    """Some files (e.g. the UCI credit-card default set) have a junk first header row
    (X1, X2, ... Y) with the REAL column names in the first data row. Detect and fix:
    if most columns are numeric below row 0 but row 0 itself is text, promote row 0 to header."""
    if len(df) < 3 or df.shape[1] < 2:
        return df
    first = df.iloc[0]
    votes = 0
    for c in df.columns:
        rest = pd.to_numeric(df[c].iloc[1:], errors="coerce")
        v0 = pd.to_numeric(pd.Series([first[c]]), errors="coerce").iloc[0]
        if rest.notna().mean() > 0.9 and pd.isna(v0) and str(first[c]).strip() != "":
            votes += 1
    if votes < max(2, df.shape[1] // 2):
        return df
    new_cols = [str(v).strip() if str(v).strip() not in ("", "nan") else f"col_{i}"
                for i, v in enumerate(first)]
    seen, uniq = {}, []                         # de-duplicate promoted names (else out[c] is a frame)
    for nm in new_cols:
        if nm in seen:
            seen[nm] += 1
            uniq.append(f"{nm}.{seen[nm]}")
        else:
            seen[nm] = 0
            uniq.append(nm)
    out = df.iloc[1:].reset_index(drop=True)
    out.columns = uniq
    for c in uniq:                              # restore numeric dtypes
        conv = pd.to_numeric(out[c], errors="coerce")
        if conv.notna().sum() >= out[c].notna().sum():
            out[c] = conv
    return out


def _promote_headers(df, max_rounds=3):
    """Apply the double-header fix repeatedly (Excel files often stack 2-3 title rows)."""
    for _ in range(max_rounds):
        new = _promote_double_header(df)
        if new is df:
            break
        df = new
    return df


def excel_to_csv(file_bytes, filename=""):
    """Convert an uploaded Excel workbook (.xlsx/.xls) to CSV text so the rest of the
    pipeline works unchanged. Uses the first NON-EMPTY sheet and trims empty rows/columns."""
    book = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
    df = None
    for name, sheet in book.items():
        sheet = sheet.dropna(how="all").dropna(axis=1, how="all")
        if len(sheet) and sheet.shape[1]:
            df = sheet
            break
    if df is None:
        raise ValueError("Every sheet in that Excel file looks empty.")
    return _promote_headers(df).to_csv(index=False)


def decode_text(file_bytes):
    """Decode raw file bytes to text: BOM sniffing, then UTF-8, then Windows-1252 (which
    never fails). Covers UTF-8/16 exports and Latin-1 European files without extra deps."""
    for bom, enc in ((b"\xff\xfe\x00\x00", "utf-32"), (b"\x00\x00\xfe\xff", "utf-32"),
                     (b"\xff\xfe", "utf-16"), (b"\xfe\xff", "utf-16"),
                     (b"\xef\xbb\xbf", "utf-8-sig")):
        if file_bytes.startswith(bom):
            return file_bytes.decode(enc), enc
    # BOM-less UTF-16 (some Windows tools): lots of NUL bytes -> decode as UTF-16, not "garbage"
    head = file_bytes[:4096]
    if head and head.count(0) > len(head) * 0.10:
        enc = "utf-16-le" if head[:1] != b"\x00" else "utf-16-be"
        try:
            return file_bytes.decode(enc), enc
        except UnicodeDecodeError:
            pass
    try:
        return file_bytes.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return file_bytes.decode("cp1252", errors="replace"), "windows-1252"


REFUSE_FORMATS = {
    ".parquet": "Parquet", ".feather": "Feather", ".sav": "SPSS (.sav)", ".rds": "R (.rds)",
    ".rdata": "R (.RData)", ".h5": "HDF5", ".hdf5": "HDF5", ".db": "SQLite", ".sqlite": "SQLite",
    ".xml": "XML", ".pdf": "PDF", ".doc": "Word", ".docx": "Word",
}


def any_to_csv(file_bytes, filename=""):
    """Turn ANY supported upload into CSV text + notes. Handles: ZIP (Kaggle downloads),
    GZIP, Excel, Stata .dta, JSON / JSON Lines, and text in UTF-8/UTF-16/Windows-1252.
    Refuses unsupported binary formats with a clear message."""
    import gzip as _gzip
    import zipfile as _zip
    notes = []
    name = (filename or "").lower()
    ext = ("." + name.rsplit(".", 1)[-1]) if "." in name else ""

    if ext in REFUSE_FORMATS and not file_bytes.startswith(b"PK"):
        raise ValueError(f"{REFUSE_FORMATS[ext]} files are not supported. Please export your "
                         f"data as CSV or Excel and upload that instead.")

    # ZIP (also .xlsx starts with PK, so check for a real archive of data files)
    if file_bytes.startswith(b"PK") and ext not in (".xlsx", ".xlsm"):
        try:
            zf = _zip.ZipFile(io.BytesIO(file_bytes))
            members = [m for m in zf.namelist()
                       if not m.endswith("/") and "__MACOSX" not in m
                       and "sample_submission" not in m.lower()
                       and m.rsplit(".", 1)[-1].lower() in
                       ("csv", "tsv", "txt", "data", "dat", "xlsx", "xls", "json", "jsonl", "dta")]
            if members:
                # prefer a 'train' file (Kaggle), else the largest data member
                members.sort(key=lambda m: zf.getinfo(m).file_size, reverse=True)
                trains = [m for m in members if "train" in m.rsplit("/", 1)[-1].lower()]
                inner = (trains or members)[0]
                if zf.getinfo(inner).file_size > 200 * 1024 * 1024:
                    raise ValueError("The file inside that ZIP is larger than 200 MB uncompressed. "
                                     "Please upload a smaller sample.")
                if len(members) > 1:
                    notes.append(f"The ZIP held {len(members)} data files; used '{inner}'.")
                else:
                    notes.append(f"Extracted '{inner}' from the ZIP archive.")
                csv, sub = any_to_csv(zf.read(inner), inner)
                return csv, notes + sub
            if ext == ".zip":
                raise ValueError("That ZIP archive has no data file (CSV/Excel/JSON) inside.")
        except _zip.BadZipFile:
            pass  # not a real zip: fall through (could be .xlsx handled below)

    if file_bytes[:2] == b"\x1f\x8b":                                    # GZIP
        inner_name = name[:-3] if name.endswith(".gz") else name
        cap = 80 * 1024 * 1024
        with _gzip.GzipFile(fileobj=io.BytesIO(file_bytes)) as g:
            data = g.read(cap + 1)                                       # bounded: no gzip-bomb OOM
        if len(data) > cap:
            raise ValueError("That GZIP unpacks to more than 80 MB. Please upload a smaller sample.")
        notes.append("Decompressed the GZIP file.")
        csv, sub = any_to_csv(data, inner_name)
        return csv, notes + sub

    if ext in (".xlsx", ".xlsm", ".xls") or file_bytes[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" \
            or (file_bytes.startswith(b"PK") and b"[Content_Types].xml" in file_bytes[:4000]):
        return excel_to_csv(file_bytes, filename), notes                 # Excel

    if ext == ".dta":                                                    # Stata
        df = pd.read_stata(io.BytesIO(file_bytes))
        notes.append("Read the Stata (.dta) file.")
        return df.to_csv(index=False), notes

    if file_bytes[:4] == b"PAR1":
        raise ValueError("Parquet files are not supported. Please export as CSV and re-upload.")

    text, enc = decode_text(file_bytes)
    if enc not in ("utf-8", "utf-8-sig"):
        notes.append(f"Decoded the file as {enc}.")

    stripped = text.lstrip()
    head = stripped[:1]
    if ext in (".json", ".jsonl") or head in ("{", "["):                 # JSON / JSON Lines
        try:
            df = None
            if head == "[":                       # a single JSON array of records
                df = pd.read_json(io.StringIO(text))
            else:                                  # JSON Lines, or fall back to one object
                try:
                    df = pd.read_json(io.StringIO(text), lines=True)
                except ValueError:
                    df = pd.read_json(io.StringIO(text))
            # a top-level object wrapping a list, e.g. {"data": [ ... ]}
            if df.shape[1] == 1 and len(df) and isinstance(df.iloc[0, 0], (dict, list)):
                inner = df.iloc[:, 0]
                if isinstance(inner.iloc[0], list) and len(df) == 1:
                    inner = pd.Series(inner.iloc[0])
                df = pd.json_normalize(inner)
            if df.shape[1] < 1 or len(df) < 1:
                raise ValueError("empty")
            notes.append("Converted the JSON file to a table.")
            return df.to_csv(index=False), notes
        except Exception:
            if ext in (".json", ".jsonl"):
                raise ValueError("Could not read that JSON. Expected an array of records "
                                 "or one JSON object per line.")
    # Excel's "sep=;" hint line
    first = text.split("\n", 1)[0].strip().lower()
    if first.startswith("sep="):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        notes.append("Applied the file's sep= hint.")
    return text, notes


# ---------------------------------------------------------------- io / profiling
def _read_arff(text):
    """KEEL / ARFF .dat files: '@attribute name ...' lines define columns, rows follow '@data'."""
    names, data_lines, in_data = [], [], False
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("%"):
            continue
        low = s.lower()
        if low.startswith("@attribute"):
            rest = s.split(None, 1)[1].strip() if len(s.split(None, 1)) > 1 else "col"
            name = rest.split("{")[0].split()[0].strip().strip("'\"") or f"col_{len(names)}"
            names.append(name)
        elif low.startswith("@data"):
            in_data = True
        elif low.startswith(("@relation", "@inputs", "@outputs", "@input", "@output")):
            continue
        elif in_data:
            data_lines.append(s)
    if not data_lines:
        raise ValueError("Found an ARFF/KEEL header but no data rows after @data.")
    body = "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(body), header=None, na_values=["?"], skipinitialspace=True)
    if names and len(names) == df.shape[1]:
        df.columns = names
    else:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    return df


def _fix_headerless(df, text, sep):
    """If every 'column name' parses as a number, the file has no header row: re-read without one."""
    hdr = [str(c) for c in df.columns]
    def _num(s):
        try:
            float(str(s).replace("Unnamed: ", "x"))
            return True
        except ValueError:
            return False
    if len(hdr) > 1 and all(_num(h) for h in hdr):
        try:
            d2 = pd.read_csv(io.StringIO(text), sep=sep, header=None, na_values=["?"],
                             engine="python" if sep == r"\s+" else "c")
            d2.columns = [f"col_{i}" for i in range(d2.shape[1])]
            return d2
        except Exception:
            pass
    return df


# Kept deliberately small: '-', '--', 'missing', 'none' are real category labels in many datasets
# (pandas' own defaults still catch 'NA'/'null'/'NaN'/'None'), so nulling them would delete classes.
NA_VALUES = ["?", "N/A", "n/a", ""]


def _try_read(text, sep, **kw):
    """One pandas read attempt. Returns (df, n_skipped_rows) or raises."""
    skipped = {"n": 0}

    def _bad(line):                       # count and skip ragged rows instead of dying
        skipped["n"] += 1
        return None
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, na_values=NA_VALUES,
                         skipinitialspace=True, engine="c", **kw)
        return df, 0
    except Exception:
        df = pd.read_csv(io.StringIO(text), sep=sep, na_values=NA_VALUES,
                         skipinitialspace=True, engine="python",
                         on_bad_lines=_bad, **kw)
        return df, skipped["n"]


_EXCEL_ERRS = ("#N/A", "#DIV/0!", "#REF!", "#VALUE!", "#NAME?", "#NUM!", "#NULL!")


def _strip_leading_comments(text):
    """Drop leading comment/preamble lines (# or //) before the table starts. A line whose first
    cell is an Excel error token (#N/A, #DIV/0! ...) is real data, not a comment."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].lstrip()
        if not s.strip():
            i += 1
            continue
        if s.startswith(("#", "//")) and not s.upper().startswith(_EXCEL_ERRS):
            i += 1
            continue
        break
    return "\n".join(lines[i:]) if 0 < i < len(lines) else text


def _skip_preamble(text, sep, warns):
    """Report exports often stack title/metadata lines above the real table. Find where the
    consistent table starts: the first row whose field count matches the modal field count
    of the rows below it. Quote-aware (uses the csv module)."""
    import csv as _csv
    delim = sep if sep in (",", ";", "\t", "|") else ","
    try:
        rows = list(_csv.reader(io.StringIO("\n".join(text.splitlines()[:40])), delimiter=delim))
    except Exception:
        return text
    counts = [len(r) for r in rows if r]
    if len(counts) < 5:
        return text
    from collections import Counter
    modal, freq = Counter(counts[2:]).most_common(1)[0]
    if modal < 2 or freq < max(3, len(counts[2:]) // 2):
        return text
    start = 0
    for idx, r in enumerate(rows):
        if len(r) == modal:
            start = idx
            break
    if start > 0:
        lines = text.splitlines()
        # map csv-row index back to a physical line (safe here: preamble rows are unquoted)
        warns.append(f"Skipped {start} preamble line(s) above the table.")
        return "\n".join(lines[start:])
    return text


def _comma_style(vals):
    """Decide what a comma means in a numeric-looking column (symbols already stripped):
    'thousands' (US 1,234 = 1234), 'decimal' (EU 1,5 = 1.5), or 'none' (no commas)."""
    import re as _re
    cvals = [v for v in vals if "," in v]
    if not cvals:
        return "none"
    both = [v for v in cvals if "." in v]
    if both:                                    # rightmost separator is the decimal point
        eu = sum(1 for v in both if v.rfind(",") > v.rfind("."))
        return "decimal" if eu >= len(both) / 2 else "thousands"
    grouped = all(_re.fullmatch(r"-?\d{1,3}(,\d{3})+", v) for v in cvals)   # proper 1,234,567
    return "thousands" if grouped else "decimal"


def _to_number(str_series, style):
    s = str_series.astype(str)
    if style == "thousands":
        s = s.str.replace(",", "", regex=False)
    elif style == "decimal":
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _clean_frame(df, warns):
    """Post-read cleanup that applies to every file."""
    # normalize column names
    df.columns = [str(c).strip() if str(c).strip() and not str(c).startswith("Unnamed:")
                  else f"col_{i}" for i, c in enumerate(df.columns)]
    # drop rows/columns that are entirely empty
    n0, c0 = df.shape
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.shape[1] < c0:
        warns.append(f"Dropped {c0 - df.shape[1]} completely empty column(s).")
    # drop a leading running-index column (0..n or 1..n)
    if df.shape[1] > 2:
        first = df.iloc[:, 0]
        if pd.api.types.is_numeric_dtype(first) and first.notna().all():
            v = first.to_numpy()
            if (v == np.arange(len(v))).all() or (v == np.arange(1, len(v) + 1)).all():
                df = df.iloc[:, 1:]
                warns.append("Dropped the first column (it was just row numbers).")
    # strip whitespace inside text cells (' positive' == 'positive')
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    # numbers dressed up as text: percent, currency, European decimals, US thousands. The comma
    # is genuinely ambiguous ('1,234' = 1234 US or 1.234 EU), so decide per column from ALL values
    # instead of a fixed rule, then convert once. This avoids the 1000x / 100x silent corruption.
    for c in df.columns:
        if df[c].dtype != object:
            continue
        raw = df[c].astype(str).str.strip()
        s = raw[raw.ne("") & raw.ne("nan")]
        if len(s) < 1:
            continue
        is_pct = s.str.contains("%").mean() > 0.8
        has_cur = s.str.contains(r"[\$€£₺]").mean() > 0.8
        core = s.str.replace(r"[%\$€£₺\s]", "", regex=True)
        if core.str.fullmatch(r"-?\d[\d.,]*").mean() <= 0.9:      # not a numeric-looking column
            continue
        style = _comma_style(core.tolist())
        if not (is_pct or has_cur or style in ("decimal", "thousands")):
            continue                                             # plain ints stay as-is
        raw_core = raw.str.replace(r"[%\$€£₺\s]", "", regex=True)
        conv = _to_number(raw_core, style)
        if conv.notna().mean() <= 0.9:
            continue
        if is_pct:
            conv = conv / 100.0
            warns.append(f"Column '{c}': converted percentages to fractions.")
        elif has_cur:
            warns.append(f"Column '{c}': removed currency symbols.")
        elif style == "decimal":
            warns.append(f"Column '{c}': converted European decimal commas to numbers.")
        else:
            warns.append(f"Column '{c}': removed thousands separators.")
        df[c] = conv
    # date/time columns: convert to days since 1970 so models can use them
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].dropna().astype(str)
            if len(s) < 5 or s.str.fullmatch(r"[\d.,\-+eE]+").mean() > 0.5:
                continue                        # numeric-ish strings are not dates
            looks_datey = s.str.contains(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}").mean() > 0.8
            if looks_datey:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    conv = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=False)
                if conv.notna().mean() > 0.9:
                    full = pd.to_datetime(df[c].astype(str), errors="coerce", format="mixed")
                    df[c] = (full - pd.Timestamp("1970-01-01")).dt.days
                    warns.append(f"Column '{c}': dates converted to day numbers.")
    # +/- infinity is not usable by the models
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        inf_ct = int(np.isinf(df[num_cols].to_numpy(dtype=float, na_value=np.nan)).sum())
        if inf_ct:
            df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
            warns.append(f"Replaced {inf_ct} infinite value(s) with missing.")
    df = df.reset_index(drop=True)
    df.attrs["read_warnings"] = warns
    return df


def read_csv(text):
    """Read ANY reasonable data file: ARFF/KEEL (.dat), delimited text with auto-detected
    separator (comma, semicolon, tab, pipe, whitespace), quoted fields, ragged rows (skipped
    with a warning), leading comments, double headers, headerless files, many NA spellings,
    European decimals, thousands separators, index columns, empty rows/columns."""
    warns = []
    text = text.lstrip("﻿\xef\xbb\xbf")
    if any(ln.strip().lower().startswith(("@relation", "@attribute"))
           for ln in text.splitlines()[:80]):
        return _clean_frame(_promote_headers(_read_arff(text)), warns)
    text = _strip_leading_comments(text)
    if not text.strip():
        raise ValueError("That file looks empty.")

    header = next((ln for ln in text.splitlines() if ln.strip()), "")
    cand = {",": header.count(","), ";": header.count(";"), "\t": header.count("\t"), "|": header.count("|")}
    sep = max(cand, key=cand.get)
    if cand[sep] == 0:
        sep = ","
    # title/metadata lines above the real table (report exports) - use the detected delimiter only,
    # so data-row commas in a semicolon file cannot masquerade as a short "preamble" header.
    t2 = _skip_preamble(text, sep, warns)
    if t2 is not text:
        text = t2
        header = next((ln for ln in text.splitlines() if ln.strip()), "")
        cand = {",": header.count(","), ";": header.count(";"),
                "\t": header.count("\t"), "|": header.count("|")}
        sep = max(cand, key=cand.get)
        if cand[sep] == 0:
            sep = ","
    df, skipped = None, 0
    try:
        df, skipped = _try_read(text, sep)
    except Exception:
        pass
    if df is None or df.shape[1] == 1:       # wrong guess: try the other delimiters
        best = (df, skipped)
        for alt in (";", "\t", "|", ","):
            if alt != sep and cand.get(alt, 0) > 0:
                try:
                    d2, s2 = _try_read(text, alt)
                    if best[0] is None or d2.shape[1] > best[0].shape[1]:
                        best, sep = (d2, s2), alt
                except Exception:
                    pass
        df, skipped = best
    if df is None or df.shape[1] == 1:       # last resort: whitespace-separated
        try:
            d3, s3 = _try_read(text, r"\s+")
            if df is None or d3.shape[1] > df.shape[1]:
                df, skipped, sep = d3, s3, r"\s+"
        except Exception:
            pass
    if df is None or df.shape[1] == 0 or len(df) == 0:
        raise ValueError("Could not find a data table in that file. Supported: CSV, Excel, "
                         "TSV, semicolon/pipe/whitespace-separated text, and ARFF/KEEL .dat.")
    if skipped:
        warns.append(f"Skipped {skipped} malformed row(s) that did not match the columns.")
    df = _fix_headerless(df, text, sep)
    return _clean_frame(_promote_headers(df), warns)


def profile(df):
    n = len(df)
    cols = []
    for c in df.columns:
        s = df[c]
        numeric = pd.api.types.is_numeric_dtype(s)
        nun = int(s.nunique(dropna=True))
        cols.append({
            "column": c,
            "dtype": "numeric" if numeric else "categorical",
            "unique": nun,
            "missing": int(s.isna().sum()),
            "id_like": bool((not numeric) and n > 0 and nun / n > ID_FRAC),
        })
    def _words(name):
        return set(str(name).strip().strip('"').lower().replace("_", " ").split())

    def _vc(name):
        return df[name].value_counts()

    def _skew(name):
        vc = _vc(name)
        return (vc.max() / max(vc.min(), 1)) if len(vc) else 1.0

    def _valid_class_col(c):
        """A plausible classification target: a small number of classes, and not an ID column.
        Numeric targets get a tighter cap because a 0-20 integer is usually a score, not classes."""
        if c["id_like"]:
            return False
        cap = NUM_CLASS_MAX if c["dtype"] == "numeric" else MAX_CLASSES
        return 2 <= c["unique"] <= cap

    hinted = [c for c in cols if _words(c["column"]) & TARGET_HINTS]
    confident, guess_reason = True, ""

    if hinted:
        # a column is NAMED like a label -> trust the name even if it is single-class or
        # continuous; prepare() then gives the right, clear refusal instead of a silent guess.
        valid_h = [c for c in hinted if _valid_class_col(c)]
        suggested = (valid_h or hinted)[-1]["column"]
    else:
        confident = False
        last, first = cols[-1], cols[0]

        def _is_cont(c):
            return c["dtype"] == "numeric" and c["unique"] > NUM_CLASS_MAX

        def _guessed(name):
            return (f"No column name looked like a label, so '{name}' was chosen automatically. "
                    f"If that is actually a feature, pick your real label with 'Wrong column?'.")

        reg_last = _is_cont(last) and bool(_words(last["column"]) & REGRESSION_HINTS)
        if _valid_class_col(last):
            suggested, guess_reason = last["column"], _guessed(last["column"])   # label at last (common)
        elif reg_last:
            suggested = last["column"]                                           # price/score last -> regression
            guess_reason = "This looks like a regression dataset (a number to predict, not classes)."
        elif _valid_class_col(first):
            suggested, guess_reason = first["column"], _guessed(first["column"])  # UCI-style label first
        elif _is_cont(last):
            suggested = last["column"]                                           # continuous last col -> regression
            guess_reason = "This looks like a regression dataset (a number to predict, not classes)."
        else:
            candidates = [c for c in cols if _valid_class_col(c)]
            if candidates:
                suggested = max(candidates, key=lambda c: _skew(c["column"]))["column"]
                guess_reason = _guessed(suggested)
            else:
                cont = [c for c in cols if _is_cont(c)]
                if cont:
                    reg = [c for c in cont if _words(c["column"]) & REGRESSION_HINTS]
                    suggested = (reg or cont)[-1]["column"]
                    guess_reason = "This looks like a regression dataset (a number to predict, not classes)."
                else:
                    suggested = last["column"]
                    guess_reason = "No clear label column was found; using the last column."
    return {"n_rows": n, "n_cols": df.shape[1], "columns": cols, "suggested_target": suggested,
            "target_confident": confident, "guess_reason": guess_reason}


def _types(X):
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return num, cat


def prepare(df, target):
    """Return X, y (0..k-1), num cols, cat cols, class labels, warnings. Raises on bad target."""
    warns = list(df.attrs.get("read_warnings", []))
    if target not in df.columns:
        raise ValueError("Target column not found.")
    n0 = len(df)
    df = df.dropna(subset=[target]).copy()
    if len(df) < n0:
        warns.append(f"Dropped {n0 - len(df)} row(s) with a missing target value.")
    y_raw = df[target]
    # normalize labels: 1.0 vs 1 (integral floats -> int), 'Yes' vs 'yes ' (case/space merge)
    if pd.api.types.is_numeric_dtype(y_raw):
        vals = y_raw.dropna()
        if len(vals) and (vals == vals.astype("int64", errors="ignore").astype(float)).all():
            try:
                y_raw = y_raw.astype("int64")
            except (TypeError, ValueError):
                pass
    else:
        s = y_raw.astype(str).str.strip()
        folded = s.str.casefold()
        if folded.nunique() < s.nunique():
            rep = s.groupby(folded).agg(lambda g: g.value_counts().index[0])
            merged = sorted(set(s.unique()) - set(rep.values))
            y_raw = folded.map(rep)
            warns.append("Merged label variants that differ only by case/spacing: "
                         + ", ".join(repr(m) for m in merged[:6]) + ".")
        else:
            y_raw = s
    df[target] = y_raw
    if pd.api.types.is_numeric_dtype(y_raw) and y_raw.nunique() > NUM_CLASS_MAX:
        raise ValueError(f"'{target}' has {y_raw.nunique()} different numeric values, so it looks like "
                         f"a number to predict (a price, score, age, measurement), not a class label. "
                         f"This tool is for classification (rare-event detection), not regression. "
                         f"If a different column holds the classes, pick it with 'Wrong column?'.")
    X = df.drop(columns=[target])
    n = len(df)
    dropped = []
    for c in list(X.columns):
        s = X[c]
        if (not pd.api.types.is_numeric_dtype(s)) and n > 0 and s.nunique() / n > ID_FRAC:
            X = X.drop(columns=[c]); dropped.append(c)
    if dropped:
        warns.append(f"Dropped ID-like / free-text column(s): {', '.join(dropped)}.")
    num, cat = _types(X)
    if not num and not cat:
        raise ValueError("No usable feature columns remain (only the target, plus any dropped "
                         "ID / free-text columns). Add at least one feature column.")
    classes = sorted(y_raw.unique(), key=lambda v: str(v))
    if len(classes) < 2:
        raise ValueError("Target has fewer than 2 classes; nothing to balance.")
    # Make the MINORITY class the positive label (index 1 for a binary target) so recall,
    # PR-AUC and ROC-AUC describe the RARE class, not the majority.
    freq = y_raw.value_counts()
    singletons = [c for c in classes if int(freq.get(c, 0)) < 2]
    if singletons:
        # a 1-row class cannot be split into train and test: drop it rather than refuse
        keep_mask = ~y_raw.isin(singletons)
        X, y_raw = X[keep_mask.values].reset_index(drop=True), y_raw[keep_mask]
        classes = [c for c in classes if c not in singletons]
        warns.append(f"Dropped class(es) with only 1 row (cannot split them into train and test): "
                     f"{', '.join(str(s) for s in singletons)}.")
        freq = y_raw.value_counts()
        if len(classes) < 2:
            raise ValueError("After removing 1-row classes, fewer than 2 classes remain. "
                             "Pick a different target column.")
    if len(classes) > MAX_CLASSES:
        raise ValueError(f"'{target}' has {len(classes)} different values. That looks like an ID, free "
                         f"text, or a multi-label column, not a class label. Pick a different column "
                         f"with 'Wrong column?'.")
    if len(classes) > 2:
        # CSRBoost and ERF-CSRBoost are two-class rare-event methods. For a multiclass target,
        # frame it the way the tool is meant for: the rarest class as the rare event vs all others.
        rare = freq.idxmin()
        pos = str(rare)
        neg = "all other classes" if pos == "other" else "other"   # the two labels must differ
        warns.append(f"Target '{target}' has {len(classes)} classes "
                     f"({', '.join(str(c) for c in sorted(classes, key=str))}). CSRBoost and "
                     f"ERF-CSRBoost are two-class methods, so this was run as a rare-event task: "
                     f"class '{rare}' (the rarest, {int(freq.min())} rows) vs all others. "
                     f"To target a different class, split your file so that class is the positive one.")
        y_raw = y_raw.apply(lambda v: pos if v == rare else neg)
        classes = [pos, neg]
        freq = y_raw.value_counts()
    classes = sorted(classes, key=lambda c: (-int(freq.get(c, 0)), str(c)))   # majority first
    y = y_raw.map({c: i for i, c in enumerate(classes)}).astype(int).values
    mino = int(pd.Series(y).value_counts().min())
    if mino < 6 or len(y) < 40:
        warns.append(f"Small dataset: the rare class has only {mino} row(s) out of {len(y)}. The "
                     f"comparison below is illustrative and will be noisy; a larger dataset gives a "
                     f"reliable answer.")
    return X.reset_index(drop=True), y, num, cat, classes, warns


def counts(y, classes):
    vc = pd.Series(y).value_counts().sort_index()
    return {str(classes[i]): int(vc.get(i, 0)) for i in range(len(classes))}


def _minority(y):
    return int(pd.Series(y).value_counts().min())


def _binary(y):
    return len(np.unique(y)) == 2


def imbalance_ratio(y):
    vc = pd.Series(y).value_counts()
    return round(float(vc.max() / max(vc.min(), 1)), 2)


def recommend(y_train, has_cat):
    ir = imbalance_ratio(y_train)
    if ir < 1.5:
        return "none", (f"Your classes are close to balanced (ratio {ir}:1). You probably do NOT "
                        f"need resampling. Try class weights first.")
    if ir < 3:
        return ("random_over" if has_cat else "smote"), (
            f"Moderate imbalance (ratio {ir}:1). Class weights often match resampling here; "
            f"a light oversample is a safe default. Compare honestly below.")
    if has_cat:
        return "random_over", (f"High imbalance (ratio {ir}:1) with categorical columns. "
                               f"Random oversampling (or SMOTE-NC) is the safe default.")
    return "smote", f"High imbalance (ratio {ir}:1) on numeric data. SMOTE is a reasonable default."


# ---------------------------------------------------------------- resampling (readable output)
def _make_sampler_original(method, num, cat, y_train, seed=SEED):
    k = max(1, min(5, _minority(y_train) - 1))
    if method == "random_over":
        return RandomOverSampler(random_state=seed)
    if method == "random_under":
        return RandomUnderSampler(random_state=seed)
    if method == "smote":
        if cat:
            idx = [i for i, c in enumerate(num + cat) if c in cat]
            return SMOTENC(categorical_features=idx, random_state=seed, k_neighbors=k)
        return SMOTE(random_state=seed, k_neighbors=k)
    if method == "borderline_smote":
        return BorderlineSMOTE(random_state=seed, k_neighbors=k)
    if method == "adasyn":
        return ADASYN(random_state=seed, n_neighbors=k)
    if method == "smote_tomek":
        return SMOTETomek(random_state=seed, smote=SMOTE(random_state=seed, k_neighbors=k))
    if method == "cluster_centroids":
        return ClusterCentroids(random_state=seed)
    return None


def _numeric_view(Xr, num, cat):
    """Scaled numeric matrix of Xr for clustering (categoricals ordinal-encoded)."""
    parts = []
    if num:
        parts.append(StandardScaler().fit_transform(Xr[num].to_numpy(dtype=float)))
    if cat:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        parts.append(enc.fit_transform(Xr[cat].astype(str)))
    return np.hstack(parts) if parts else np.zeros((len(Xr), 1))


def _clustered_retain(enc_maj, pct, K, seed):
    """CSRBoost undersampling: MiniBatchKMeans on the majority (K = n_minority), keep pct of
    each cluster. Returns the indices (into enc_maj) of the majority rows to keep.
    Same clustering parameters as the research ERF script (mini-batch KMeans, not full KMeans)."""
    K = max(1, min(K, len(enc_maj)))
    rng = np.random.RandomState(seed)
    labels = MiniBatchKMeans(n_clusters=K, n_init=1, max_iter=100,
                             batch_size=1024, random_state=seed).fit_predict(enc_maj)
    kept = []
    for k in range(K):
        cidx = np.where(labels == k)[0]
        n_keep = max(1, math.ceil(len(cidx) * pct))
        kept.extend(cidx.tolist() if n_keep >= len(cidx)
                    else rng.choice(cidx, n_keep, replace=False).tolist())
    return np.array(kept, dtype=int)


def _apply_sampler_original(sampler, Xr, y, num, cat, cols):
    """Apply an imblearn sampler in the ORIGINAL feature space, keeping the CSV readable."""
    if isinstance(sampler, SMOTENC):
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xenc = Xr.copy(); Xenc[cat] = enc.fit_transform(Xr[cat])
        Xres, yres = sampler.fit_resample(Xenc.values, y)
        Xres = pd.DataFrame(Xres, columns=cols)
        Xres[cat] = enc.inverse_transform(Xres[cat])
        for c in num:
            Xres[c] = pd.to_numeric(Xres[c], errors="coerce")
    else:
        Xres, yres = sampler.fit_resample(Xr, y)
        Xres = pd.DataFrame(Xres, columns=cols)
    return Xres.reset_index(drop=True), np.asarray(yres)


def resample_train(method, X_train, y_train, num, cat, seed=SEED):
    """Resample the TRAIN split in the original feature space (human-readable). Returns
    (X_res_df, y_res, note). Never sees the test set. `seed` varies per ERF-CSRBoost bag."""
    if method == "none":
        return X_train.copy(), y_train, "No resampling applied (class weights recommended instead)."
    cols = num + cat
    Xr = X_train[cols].copy()
    for c in num:
        Xr[c] = Xr[c].fillna(Xr[c].median())
    for c in cat:
        mode = Xr[c].mode()
        Xr[c] = Xr[c].fillna(mode.iloc[0] if not mode.empty else "missing").astype(str)
    y = np.asarray(y_train)
    note = ""

    # CSRBoost method: cluster the majority (K = n_minority), keep 50% per cluster, then SMOTE.
    if method == "clustered_smote":
        if len(np.unique(y)) != 2:
            note = "Clustered undersampling needs exactly 2 classes; used SMOTE instead. "
            method = "smote"
        else:
            maj = np.where(y == 0)[0]; mino = np.where(y == 1)[0]
            enc = _numeric_view(Xr, num, cat)
            K = max(1, min(len(mino), len(maj)))
            keep = maj[_clustered_retain(enc[maj], 0.50, K, seed)]
            sel = np.concatenate([mino, keep])
            Xr = Xr.iloc[sel].reset_index(drop=True); y = y[sel]
            note = ("Clustered undersampling + SMOTE-NC (the CSRBoost method). " if cat
                    else "Clustered undersampling + SMOTE (the CSRBoost method). ")
            method = "smote"

    m = method
    if cat and m in NUMERIC_ONLY:
        note += f"'{m}' needs all-numeric data; used random oversampling for the mixed-type columns. "
        m = "random_over"
    if cat and m == "smote" and "CSRBoost" not in note:
        note += "Categorical columns present, used SMOTE-NC. "
    sampler = _make_sampler_original(m, num, cat, y, seed)
    try:
        if not cat and m not in ("random_over", "random_under"):
            # Distance-based samplers (SMOTE etc.) run in STANDARDIZED space
            # (StandardScaler fit on train, then resample).
            # Results are inverse-transformed back to original units so the CSV stays readable.
            scaler = StandardScaler().fit(Xr[num])
            Xres_s, yres = sampler.fit_resample(scaler.transform(Xr[num]), y)
            Xres = pd.DataFrame(scaler.inverse_transform(Xres_s), columns=num)
            yres = np.asarray(yres)
        else:
            Xres, yres = _apply_sampler_original(sampler, Xr, y, num, cat, cols)
    except Exception as e:                       # never crash the demo: fall back safely
        Xres, yres = _apply_sampler_original(RandomOverSampler(random_state=seed), Xr, y, num, cat, cols)
        note += f"(fell back to random oversampling: {e}) "
    return Xres, yres, note.strip()


# ---------------------------------------------------------------- the three compared models
from models import make_plain, csrboost_fit_predict, train_erf as _train_erf_impl  # noqa: E402


def train_erf(X_train, y_train, num, cat):
    """ERF-CSRBoost (experimental, unpublished extension of CSRBoost), balancing injected from this engine."""
    return _train_erf_impl(X_train, y_train, num, cat, resample_train, _preprocessor)


class _ProbaModel:
    """Wraps precomputed test-set probabilities so _score/_binary logic can reuse them."""
    def __init__(self, probs):
        self._p = np.asarray(probs, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1.0 - self._p, self._p])

    def predict(self, X):
        return (self._p >= 0.5).astype(int)


# ---------------------------------------------------------------- classify: does it help?
def _preprocessor(num, cat):
    parts = []
    if num:
        parts.append(("num", SkPipeline([("imp", SimpleImputer(strategy="median")),
                                         ("sc", StandardScaler())]), num))
    if cat:
        parts.append(("cat", SkPipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                         ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat))
    return ColumnTransformer(parts)


def _clf(name, weighted=False):
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=SEED,
                                      class_weight="balanced" if weighted else None)
    if name == "adaboost":
        return AdaBoostClassifier(random_state=SEED)       # no class_weight param
    return LogisticRegression(max_iter=1000,
                              class_weight="balanced" if weighted else None)


def _metrics(y_true, y_pred, y_score):
    avg = "binary" if _binary(y_true) else "macro"
    out = {
        "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "gmean": geometric_mean_score(y_true, y_pred, average=avg),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if _binary(y_true) and y_score is not None:
        try:
            out["pr_auc"] = average_precision_score(y_true, y_score)
            out["roc_auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            pass
    return {k: round(float(v), 4) for k, v in out.items()}


def _score(model, X_test, y_test):
    yp = model.predict(X_test)
    ys = None
    if _binary(y_test) and hasattr(model, "predict_proba"):
        try:
            ys = model.predict_proba(X_test)[:, 1]
        except Exception:
            ys = None
    return _metrics(y_test, yp, ys)


CAP_COMPARE = 4000     # training rows for the 3-way model comparison (keeps the demo live)


def _cap_train(X, y, cap=CAP_COMPARE, seed=SEED):
    if len(y) <= cap:
        return X, y, False
    idx = _proba_subsample(y, cap, seed)
    return X.iloc[idx].reset_index(drop=True), np.asarray(y)[idx], True


def _dense(m):
    return m.toarray() if hasattr(m, "toarray") else np.asarray(m, dtype=float)


def classify_compare(method, classifier, X_train, X_test, y_train, y_test, num, cat):
    """The THREE-WAY honest comparison on the UNTOUCHED test set:
      plain    : standard RandomForest + class weights on the original data (the one-liner)
      csrboost : the published method, as replicated for this project (a port, not the original authors' code)
      erf      : ERF-CSRBoost, an experimental, unpublished extension of CSRBoost
    The three do not have equal capacity (plain: one 200-tree forest; csrboost: AdaBoost of 50 trees;
    erf: 1,000 trees), so a difference between them is not a controlled test of the method.
    All three are judged on the same untouched test rows. Training is stratified-capped on very
    large uploads so the live demo stays responsive (noted to the user)."""
    if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
        raise ValueError("The rare class has too few rows to appear in both the training split and the "
                         "held-out test split, so no honest comparison is possible. Use a file with at "
                         "least a handful of rare-class rows (about 6 or more).")
    Xtr, ytr, capped = _cap_train(X_train, y_train)
    note = (f"Models trained on a stratified {CAP_COMPARE:,}-row sample of your training split "
            f"for responsiveness; the downloadable balanced file uses ALL rows." if capped else "")

    plain_pipe = SkPipeline([("pre", _preprocessor(num, cat)), ("clf", make_plain(weighted=True))])
    plain_pipe.fit(Xtr, ytr)
    plain = _score(plain_pipe, X_test, y_test)

    if not _binary(y_test):
        # multiclass: CSRBoost/ERF are binary methods; compare plain vs balanced-plain instead
        m = method if method != "none" else "smote"
        Xres, yres, _ = resample_train(m, Xtr, ytr, num, cat)
        bal = SkPipeline([("pre", _preprocessor(num, cat)), ("clf", make_plain(weighted=False))]).fit(Xres, yres)
        return {"plain": plain, "csrboost": None, "erf": _score(bal, X_test, y_test),
                "proba": None, "detector": None, "erf_is_fallback": True,
                "note": (note + " Multiclass target: showing plain vs balanced comparison "
                                "(CSRBoost and ERF-CSRBoost are two-class methods).").strip()}

    csr, erf, proba, detector = None, None, None, None
    try:
        # CSRBoost (exact replication) operates on preprocessed numeric matrices
        pre = _preprocessor(num, cat).fit(Xtr)
        csr_probs = csrboost_fit_predict(_dense(pre.transform(Xtr)), np.asarray(ytr),
                                         _dense(pre.transform(X_test)), SEED)
        csr_model = _ProbaModel(csr_probs)
        csr = _score(csr_model, X_test, y_test)
    except Exception as e:
        note = (note + f" CSRBoost could not run on this data ({e}).").strip()

    try:
        erf_model = train_erf(Xtr, ytr, num, cat)
        erf = _score(erf_model, X_test, y_test)
        idx = _proba_subsample(y_test, 5000)
        proba = {
            "y_test": [int(v) for v in np.asarray(y_test)[idx]],
            "plain": [round(float(p), 4) for p in plain_pipe.predict_proba(X_test.iloc[idx])[:, 1]],
            "csrboost": ([round(float(csr_probs[i]), 4) for i in idx] if csr is not None else None),
            "erf": [round(float(p), 4) for p in erf_model.predict_proba(X_test.iloc[idx])[:, 1]],
            "subsampled": bool(len(idx) < len(y_test)),
        }
        detector = build_detector(erf_model, X_test, y_test, num, cat)
    except Exception as e:
        note = (note + f" Could not train ERF-CSRBoost ({e}).").strip()

    return {"plain": plain, "csrboost": csr, "erf": erf,
            "proba": proba, "detector": detector, "note": note}


def _fmt_val(v):
    try:
        f = float(v)
        return int(f) if f == int(f) else round(f, 2)
    except (ValueError, TypeError):
        return str(v)


def build_detector(clf_pipe, X_test, y_test, num, cat):
    """Product B: the trained detector applied to the UNSEEN test rows, ranked by risk."""
    probs = clf_pipe.predict_proba(X_test)[:, 1]
    y = np.asarray(y_test)
    flagged = int((probs >= 0.5).sum())
    caught = int(((probs >= 0.5) & (y == 1)).sum())
    total_rare = int((y == 1).sum())
    disp = (num[:4] if num else cat[:4])
    order = np.argsort(-probs)[:15]
    rows = [{"prob": round(float(probs[i]), 4), "actual": int(y[i]),
             "cols": {c: _fmt_val(X_test.iloc[int(i)][c]) for c in disp}} for i in order]
    return {"flagged": flagged, "caught": caught, "missed": total_rare - caught,
            "total_rare": total_rare, "total": int(len(y)), "display_cols": disp, "rows": rows}


def score_new(train_csv, target, method, classifier, score_csv):
    """Product B upload path: train a detector on ALL labeled rows (balanced), then flag
    the user's new, unlabeled transactions."""
    Xtr, ytr, num, cat, classes, _ = prepare(read_csv(train_csv), target)
    if classifier == "erf" and len(classes) == 2:
        pipe = train_erf(Xtr, ytr, num, cat)          # ERF-CSRBoost (experimental extension)
    else:
        base_name = "random_forest" if classifier == "erf" else classifier
        Xb, yb, _ = resample_train(method, Xtr, ytr, num, cat)
        pipe = SkPipeline([("pre", _preprocessor(num, cat)), ("clf", _clf(base_name))]).fit(Xb, yb)
    sdf = read_csv(score_csv)
    feat = sdf.drop(columns=[target]) if target in sdf.columns else sdf
    missing = [c for c in (num + cat) if c not in feat.columns]
    if missing:
        raise ValueError("Your file is missing these columns: " + ", ".join(missing))
    Xs = feat[num + cat]
    # classes are majority-first, so the rarest (positive) class is the LAST proba column;
    # [:, -1] keeps risk aligned with pos for multiclass too (identical to [:, 1] when binary).
    probs = pipe.predict_proba(Xs)[:, -1]
    pos = str(classes[-1])
    out = sdf.copy()
    out[f"{target}_risk"] = [round(float(p), 4) for p in probs]
    out["flag"] = ["FLAG" if p >= 0.5 else "" for p in probs]
    disp = (num[:4] if num else cat[:4])
    preview = [{"prob": round(float(probs[i]), 4), "flag": bool(probs[i] >= 0.5),
                "cols": {c: _fmt_val(Xs.iloc[i][c]) for c in disp}}
               for i in np.argsort(-probs)[:15]]
    return {"n": int(len(probs)), "flagged": int((probs >= 0.5).sum()), "positive_label": pos,
            "target": target, "display_cols": disp, "preview": preview,
            "scored_csv": out.to_csv(index=False)}


def _proba_subsample(y, cap, seed=SEED):
    y = np.asarray(y)
    if len(y) <= cap:
        return np.arange(len(y))
    rng = np.random.RandomState(seed)
    keep = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        # keep the proportional share, but never shrink a class below 60 rows (or its full size);
        # otherwise an extreme minority collapses to 1 and stratified k-fold later crashes.
        k = max(round(cap * len(idx) / len(y)), min(len(idx), 60))
        keep.extend(rng.choice(idx, min(k, len(idx)), replace=False))
    return np.array(sorted(keep))


def leakage_lab(method, X, y, num, cat):
    """Teaching demo of why 'balance everything, then split' inflates the score. Uses a plain
    RandomForest for BOTH arms so the two numbers are directly comparable (independent of the
    headline ERF-CSRBoost model)."""
    try:
        def _rf():
            return SkPipeline([("pre", _preprocessor(num, cat)),
                               ("clf", RandomForestClassifier(n_estimators=120, random_state=SEED, n_jobs=-1))])
        if len(y) > 6000:                     # keep the teaching demo fast on big uploads
            idx = _proba_subsample(y, 6000)
            X, y = X.iloc[idx].reset_index(drop=True), np.asarray(y)[idx]
        # honest: split first, resample TRAIN only, score on the untouched test set
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
        Xb, yb, _ = resample_train(method, Xtr, ytr, num, cat)
        honest = _score(_rf().fit(Xb, yb), Xte, yte)
        # leaky: resample the WHOLE dataset, THEN split (synthetic test twins leak into training)
        Xr, yr, _ = resample_train(method, X, y, num, cat)
        Xtr2, Xte2, ytr2, yte2 = train_test_split(Xr, yr, test_size=TEST_SIZE, stratify=yr, random_state=SEED)
        leaky = _score(_rf().fit(Xtr2, ytr2), Xte2, yte2)
        return {
            "leaky": {k: leaky[k] for k in ("balanced_accuracy", "recall", "f1") if k in leaky},
            "honest": {k: honest[k] for k in ("balanced_accuracy", "recall", "f1") if k in honest},
            "delta_balanced_accuracy": round(leaky["balanced_accuracy"] - honest["balanced_accuracy"], 4),
        }
    except Exception:
        return None


def leakage_scan(X, y, cap=4000):
    """Two more leak checks beyond balance-before-split:
      - duplicate rows: identical feature rows that leak if they land on both sides of a split.
      - proxy feature: a single column that predicts the label almost perfectly (a leaked or
        target-derived column). Runs on a capped sample for speed."""
    out = {"dupes": 0, "dupes_pct": 0.0, "proxies": []}
    try:
        Xs, ys = X, np.asarray(y)
        if len(Xs) > cap:
            idx = _proba_subsample(ys, cap)
            Xs, ys = Xs.iloc[idx].reset_index(drop=True), ys[idx]
        n = len(Xs)
        dup = int(Xs.duplicated().sum())
        out["dupes"], out["dupes_pct"] = dup, round(100.0 * dup / max(n, 1), 1)
        if len(np.unique(ys)) == 2:
            for c in Xs.columns:
                try:
                    s = Xs[c]
                    if pd.api.types.is_numeric_dtype(s):
                        v = pd.to_numeric(s, errors="coerce")
                        m = v.notna().values
                        if m.sum() < 20 or v[m].nunique() < 2:
                            continue
                        score, yy = v[m].values, ys[m]
                    else:
                        codes = s.astype("category").cat.codes
                        m = (codes >= 0).values
                        if m.sum() < 20 or codes[m].nunique() < 2:
                            continue
                        # order categories by their class-1 rate, then score by that rate
                        tmp = pd.DataFrame({"c": codes[m].values, "y": ys[m]})
                        score, yy = tmp.groupby("c")["y"].transform("mean").values, ys[m]
                    a = roc_auc_score(yy, score)
                    a = max(a, 1.0 - a)                 # direction-agnostic
                    if a >= 0.985:
                        out["proxies"].append({"column": str(c), "auc": round(float(a), 3)})
                except Exception:
                    continue
            out["proxies"].sort(key=lambda p: -p["auc"])
            out["proxies"] = out["proxies"][:5]
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- the hero: pipeline snippet
_IMPORTS = {
    "none": "",
    "random_over": "from imblearn.over_sampling import RandomOverSampler",
    "random_under": "from imblearn.under_sampling import RandomUnderSampler",
    "smote": "from imblearn.over_sampling import SMOTE",
    "borderline_smote": "from imblearn.over_sampling import BorderlineSMOTE",
    "adasyn": "from imblearn.over_sampling import ADASYN",
    "smote_tomek": "from imblearn.combine import SMOTETomek",
    "cluster_centroids": "from imblearn.under_sampling import ClusterCentroids",
}
_CTOR = {
    "none": None, "random_over": "RandomOverSampler(random_state=42)",
    "random_under": "RandomUnderSampler(random_state=42)", "smote": "SMOTE(random_state=42)",
    "borderline_smote": "BorderlineSMOTE(random_state=42)", "adasyn": "ADASYN(random_state=42)",
    "smote_tomek": "SMOTETomek(random_state=42)", "cluster_centroids": "ClusterCentroids(random_state=42)",
}
_CLF_CTOR = {
    "logreg": ("from sklearn.linear_model import LogisticRegression", "LogisticRegression(max_iter=1000)"),
    "random_forest": ("from sklearn.ensemble import RandomForestClassifier", "RandomForestClassifier(n_estimators=200, random_state=42)"),
    "adaboost": ("from sklearn.ensemble import AdaBoostClassifier", "AdaBoostClassifier(random_state=42)"),
}


def _clustered_snippet(classifier, target):
    if classifier == "erf":
        clf_imp = ""
        train_lines = ["# 3) Train the model on the balanced data.",
                       "#    (the app also trained ERF-CSRBoost here, an experimental ensemble of",
                       "#    resampled random forests; its code is in backend/models/erf_csrboost.py",
                       "#    of the project repository and it is not a pip package)",
                       "# clf = YourClassifier().fit(X_bal, y_bal)"]
    else:
        clf_imp, clf_ctor = _CLF_CTOR.get(classifier, _CLF_CTOR["logreg"])
        train_lines = ["# 3) Train, then score on the UNTOUCHED test set.",
                       f"clf = {clf_ctor}.fit(X_bal, y_bal)",
                       "print(clf.score(X_test, y_test))"]
    return "\n".join([
        "# Leakage-safe CSRBoost balancing: everything below runs on the TRAIN split only.",
        "#",
        "# Assumptions of this snippet (check them against your own data):",
        "#  - `df` is a pandas DataFrame you have already loaded.",
        "#  - the target is coded 0/1 with 1 = the rare class. The app picks the rarest class as 1",
        "#    for you; here you must recode string labels yourself, or the clustering step below",
        "#    will look at the wrong class.",
        "#  - only numeric feature columns are kept (categorical columns are dropped), and they must",
        "#    contain no missing values. Impute after the split, fitted on the train split only.",
        "#  - this illustrates the balancing step. It is not the exact code the app ran: the app also",
        "#    imputes and one-hot encodes, fitted on the train split.",
        "import math, numpy as np",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.cluster import MiniBatchKMeans",
        "from sklearn.preprocessing import StandardScaler",
        "from imblearn.over_sampling import SMOTE",
        clf_imp,
        "",
        f"X, y = df.drop(columns=[{target!r}]).select_dtypes('number'), df[{target!r}]",
        "X_train, X_test, y_train, y_test = train_test_split(",
        "    X, y, test_size=0.25, stratify=y, random_state=42)",
        "",
        "# 1) Cluster the MAJORITY (K = number of minority rows), keep 50% of each cluster.",
        "Xs = StandardScaler().fit_transform(X_train)",
        "maj = np.where(y_train == 0)[0]; mino = np.where(y_train == 1)[0]",
        "K = max(1, min(len(mino), len(maj)))",
        "labels = MiniBatchKMeans(n_clusters=K, n_init=1, random_state=42).fit_predict(Xs[maj])",
        "rng, keep = np.random.RandomState(42), []",
        "for k in range(K):",
        "    idx = maj[labels == k]",
        "    n = max(1, math.ceil(len(idx) * 0.5))",
        "    keep += list(idx) if n >= len(idx) else list(rng.choice(idx, n, replace=False))",
        "sel = np.concatenate([mino, keep])",
        "X_u, y_u = X_train.iloc[sel], y_train.iloc[sel]",
        "",
        "# 2) SMOTE the minority up on the undersampled set, in STANDARDIZED space",
        "#    (SMOTE interpolates by distance, so scale first; invert after if you need raw units).",
        "sc2 = StandardScaler().fit(X_u)",
        "X_bal_s, y_bal = SMOTE(random_state=42).fit_resample(sc2.transform(X_u), y_u)",
        "X_bal = sc2.inverse_transform(X_bal_s)",
        "",
        *train_lines,
    ])


def pipeline_snippet(method, classifier, target, num, cat):
    if method == "clustered_smote":
        return _clustered_snippet(classifier, target)
    clf_imp, clf_ctor = _CLF_CTOR.get(classifier, _CLF_CTOR["logreg"])
    res_imp = _IMPORTS.get(method, "")
    res_ctor = _CTOR.get(method)
    lines = [
        "# Leakage-safe: this Pipeline resamples ONLY the training folds and never the test set.",
        "# Assumes `df` is a pandas DataFrame you have already loaded.",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.pipeline import Pipeline as Pre",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder",
        "from imblearn.pipeline import Pipeline",
        clf_imp,
    ]
    if res_imp:
        lines.append(res_imp)
    lines += [
        "",
        f"target = {target!r}",
        f"num = {num!r}",
        f"cat = {cat!r}",
        "X, y = df.drop(columns=[target]), df[target]",
        "",
        "# 1) SPLIT FIRST, before any resampling.",
        "X_train, X_test, y_train, y_test = train_test_split(",
        "    X, y, test_size=0.25, stratify=y, random_state=42)",
        "",
        "pre = ColumnTransformer([",
        "    ('num', Pre([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num),",
        "    ('cat', Pre([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore'))]), cat),",
        "])",
        "",
        "pipe = Pipeline([",
        "    ('pre', pre),",
    ]
    if res_ctor:
        lines.append(f"    ('resampler', {res_ctor}),   # runs on TRAIN folds only")
    lines += [
        f"    ('clf', {clf_ctor}),",
        "])",
        "",
        "pipe.fit(X_train, y_train)          # resampling happens inside fit, on train only",
        "print(pipe.score(X_test, y_test))   # test set stays untouched and imbalanced",
    ]
    return "\n".join(l for l in lines if l is not None)


# ---------------------------------------------------------------- full run
def cross_val_compare(X, y, num, cat, folds):
    """Out-of-fold 3-way comparison (plain / CSRBoost / ERF-CSRBoost) over repeated stratified
    folds, in the spirit of the research 100-fold protocol. Averages away the single-split noise. Each sample
    is predicted only when it is in a held-out fold, so it is always leakage-safe. USER UPLOADS only
    (the demos keep the fast single split)."""
    from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
    y = np.asarray(y)
    if not _binary(y):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
        r = classify_compare("clustered_smote", "erf", Xtr, Xte, ytr, yte, num, cat)
        r["folds"] = 1
        return r

    caps = {3: 8000, 20: 4000, 100: 1500}
    Xc, yc, capped = _cap_train(X, y, caps.get(folds, 4000))
    n = len(yc)
    n_splits = max(2, folds) if folds <= 3 else 5
    if int(pd.Series(yc).value_counts().min()) < n_splits:
        # too few minority rows to stratify into this many folds: fall back to one honest split
        Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=TEST_SIZE, stratify=yc, random_state=SEED)
        r = classify_compare("clustered_smote", "erf", Xtr, Xte, ytr, yte, num, cat)
        r["folds"] = 1
        r["folds_downgraded_from"] = int(folds)   # the UI says so; the rare class is too small to fold
        return r
    if folds <= 3:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    else:
        splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=max(1, round(folds / 5)), random_state=SEED)
    total = splitter.get_n_splits(Xc, yc)

    sp, sc, se, cnt = (np.zeros(n) for _ in range(4))
    csr_ok, wins, note = True, 0, ""
    for tr, te in splitter.split(Xc, yc):
        Xtr, Xte, ytr, yte = Xc.iloc[tr], Xc.iloc[te], yc[tr], yc[te]
        pre = _preprocessor(num, cat).fit(Xtr)
        pp = SkPipeline([("pre", _preprocessor(num, cat)), ("clf", make_plain(True))]).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        try:
            cp = csrboost_fit_predict(_dense(pre.transform(Xtr)), ytr, _dense(pre.transform(Xte)), SEED)
        except Exception:
            csr_ok, cp = False, None
        ep = train_erf(Xtr, ytr, num, cat).predict_proba(Xte)[:, 1]
        sp[te] += pp; se[te] += ep; cnt[te] += 1
        if csr_ok and cp is not None:
            sc[te] += cp
            try:
                if roc_auc_score(yte, ep) > roc_auc_score(yte, cp):
                    wins += 1
            except Exception:
                pass
    denom = np.maximum(cnt, 1)
    pool_p, pool_e, pool_c = sp / denom, se / denom, sc / denom

    def M(prob):
        return _metrics(yc, (prob >= 0.5).astype(int), prob)

    plain_m, erf_m = M(pool_p), M(pool_e)
    csr_m = M(pool_c) if csr_ok else None
    idx = _proba_subsample(yc, 5000)
    proba = {"y_test": [int(v) for v in yc[idx]],
             "plain": [round(float(pool_p[i]), 4) for i in idx],
             "csrboost": ([round(float(pool_c[i]), 4) for i in idx] if csr_ok else None),
             "erf": [round(float(pool_e[i]), 4) for i in idx],
             "subsampled": bool(len(idx) < n)}
    detector = build_detector(_ProbaModel(pool_e), Xc.reset_index(drop=True), yc, num, cat)
    if capped:
        note = (f"Used a stratified {len(yc):,}-row sample so the {total}-fold run stays responsive.")
    return {"plain": plain_m, "csrboost": csr_m, "erf": erf_m, "proba": proba, "detector": detector,
            "folds": total, "wins_erf_vs_csr": wins, "note": note}


def analyze(csv_text, target, method, do_classify=False, classifier="logreg", folds=1):
    df = read_csv(csv_text)
    X, y, num, cat, classes, warns = prepare(df, target)
    rec_method, rec_msg = recommend(y, bool(cat))

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    before = counts(ytr, classes)
    Xres, yres, note = resample_train(method, Xtr, ytr, num, cat)
    after = counts(yres, classes)

    train_out = Xres.copy(); train_out[target] = [classes[i] for i in yres]
    test_out = Xte.copy(); test_out[target] = [classes[i] for i in yte]

    result = {
        "ok": True,
        "target": target, "num": num, "cat": cat,
        "n_rows": len(y), "ir": imbalance_ratio(y),
        "positive_label": str(classes[-1]), "negative_label": str(classes[0]),
        "full_counts": counts(y, classes),
        "warnings": warns, "method_note": note,
        "recommend": {"method": rec_method, "message": rec_msg},
        "before_train": before, "after_train": after,
        "test_counts": counts(yte, classes),
        "snippet": pipeline_snippet(method, classifier, target, num, cat),
        "balanced_train_csv": train_out.to_csv(index=False),
        "test_untouched_csv": test_out.to_csv(index=False),
    }
    if do_classify:
        if folds and folds > 1:
            eff = folds                      # large uploads: cap folds so the live run stays responsive
            if len(y) > 30000 and eff > 20:
                eff = 20
            if len(y) > 100000 and eff > 3:
                eff = 3
            if eff != folds:
                warns.append(f"Large dataset ({len(y):,} rows): capped the live check at {eff} folds to "
                             f"stay responsive. The downloadable code runs the full protocol you pick.")
            result["classify"] = cross_val_compare(X, y, num, cat, eff)
        else:
            result["classify"] = classify_compare(method, classifier, Xtr, Xte, ytr, yte, num, cat)
        if result["classify"].get("erf"):
            result["leakage_lab"] = leakage_lab("clustered_smote", X, y, num, cat)
            result["leakage_scan"] = leakage_scan(X, y)
    return result


if __name__ == "__main__":
    import sys, json
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        txt = open(path, encoding="utf-8").read()
        df = read_csv(txt)
        p = profile(df)
        tgt = p["suggested_target"]
        print("PROFILE:", p["n_rows"], "rows,", p["n_cols"], "cols; target=", tgt)
        for m in ["none", "random_over", "smote"]:
            r = analyze(txt, tgt, m, do_classify=True, classifier="logreg")
            print(f"\n[{m}] before={r['before_train']} after={r['after_train']} "
                  f"test={r['test_counts']} note={r['method_note']!r}")
            if "classify" in r:
                print("  baseline:", r["classify"]["baseline_weighted"])
                print("  resample:", r["classify"]["resampled"])
