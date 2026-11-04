"""Torture test: every realistic messy file must either process cleanly or fail with a
clear, human-readable message. Run: py -3.11 backend/test_torture.py"""
import io
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, "backend")
import engine  # noqa: E402

rng = np.random.RandomState(7)


def base_rows(n=300, rare=0.15, cols=3, sep=",", header=True, label=("0", "1")):
    lines = []
    if header:
        lines.append(sep.join([f"f{i}" for i in range(cols)] + ["target"]))
    for i in range(n):
        y = label[1] if rng.rand() < rare else label[0]
        vals = [f"{rng.randn() + (1.2 if y == label[1] else 0):.3f}" for _ in range(cols)]
        lines.append(sep.join(vals + [y]))
    return lines


CASES = {}

# 1 ragged rows (the reported bug): some rows have EXTRA fields
r = base_rows()
r[8] = r[8] + ",junk,junk"
r[15] = r[15] + ",x"
CASES["ragged_extra_fields"] = "\n".join(r)

# 2 ragged rows: some rows have MISSING fields
r = base_rows()
r[5] = ",".join(r[5].split(",")[:2])
CASES["ragged_missing_fields"] = "\n".join(r)

# 3 semicolon delimited
CASES["semicolon"] = "\n".join(base_rows(sep=";"))

# 4 tab delimited
CASES["tab"] = "\n".join(base_rows(sep="\t"))

# 5 pipe delimited
CASES["pipe"] = "\n".join(base_rows(sep="|"))

# 6 whitespace (UCI .data style), no header
CASES["whitespace_noheader"] = "\n".join(base_rows(sep=" ", header=False))

# 7 headerless comma file
CASES["comma_noheader"] = "\n".join(base_rows(header=False))

# 8 leading comment lines
CASES["leading_comments"] = "# exported by tool v1.2\n# date: 2026\n\n" + "\n".join(base_rows())

# 9 UTF-8 BOM
CASES["utf8_bom"] = "﻿" + "\n".join(base_rows())

# 10 quoted fields containing commas
r = ["name,desc,target"]
for i in range(200):
    y = "1" if rng.rand() < 0.15 else "0"
    r.append(f'item{i},"has, commas, inside",{y}')
CASES["quoted_commas"] = "\n".join(r)

# 11 many NA spellings
r = base_rows()
for i in (3, 7, 11, 19):
    parts = r[i].split(","); parts[0] = ["NA", "N/A", "null", "?"][i % 4]; r[i] = ",".join(parts)
CASES["na_spellings"] = "\n".join(r)

# 12 European decimal commas with semicolon sep
r = ["f0;f1;target"]
for i in range(250):
    y = "1" if rng.rand() < 0.15 else "0"
    r.append(f"{rng.randn()+ (1.2 if y=='1' else 0):.3f};{rng.randn():.3f}".replace(".", ",") + f";{y}")
CASES["euro_decimals"] = "\n".join(r)

# 13 thousands separators
r = ["amount,age,target"]
for i in range(250):
    y = "1" if rng.rand() < 0.15 else "0"
    amt = int(abs(rng.randn()) * 5000 + (3000 if y == "1" else 0)) + 1000
    r.append(f'"{amt:,}",{rng.randint(18,80)},{y}')
CASES["thousands_sep"] = "\n".join(r)

# 14 running index column
r = ["idx,f0,f1,target"]
for i in range(250):
    y = "1" if rng.rand() < 0.15 else "0"
    r.append(f"{i},{rng.randn()+(1.2 if y=='1' else 0):.3f},{rng.randn():.3f},{y}")
CASES["index_column"] = "\n".join(r)

# 15 empty rows sprinkled in
r = base_rows()
r.insert(10, ""); r.insert(50, "   ")
CASES["blank_lines"] = "\n".join(r)

# 16 trailing commas -> phantom empty column
CASES["trailing_commas"] = "\n".join(ln + "," for ln in base_rows())

# 17 duplicate column names
r = base_rows()
r[0] = "f,f,f,target"
CASES["dup_colnames"] = "\n".join(r)

# 18 whitespace around values and labels
r = ["f0, f1 , target "]
for i in range(250):
    y = " positive " if rng.rand() < 0.15 else "negative"
    r.append(f" {rng.randn()+(1.2 if 'pos' in y else 0):.3f} , {rng.randn():.3f} ,{y}")
CASES["padded_values"] = "\n".join(r)

# 19 boolean-looking target
r = ["f0,f1,churned"]
for i in range(250):
    y = "True" if rng.rand() < 0.15 else "False"
    r.append(f"{rng.randn()+(1.2 if y=='True' else 0):.3f},{rng.randn():.3f},{y}")
CASES["bool_target"] = "\n".join(r)

# 20 double header (UCI credit-card style)
r = ["c0,X1,X2,Y", "ID,LIMIT,AGE,default payment next month"]
for i in range(250):
    y = "1" if rng.rand() < 0.2 else "0"
    r.append(f"{i+1},{rng.randint(1000,90000)},{rng.randint(18,70)},{y}")
CASES["double_header"] = "\n".join(r)

# 21 ARFF / KEEL
arff = ["@relation toy", "@attribute A real", "@attribute B real",
        "@attribute Class {positive, negative}", "@data"]
for i in range(250):
    y = "positive" if rng.rand() < 0.15 else "negative"
    arff.append(f"{rng.randn()+(1.2 if y=='positive' else 0):.3f}, {rng.randn():.3f}, {y}")
CASES["arff_keel"] = "\n".join(arff)

# 22 mixed types + missing values everywhere
r = ["city,amount,age,target"]
for i in range(300):
    y = "1" if rng.rand() < 0.15 else "0"
    city = rng.choice(["Berlin", "Munich", "?", "Hamburg"])
    amt = "" if rng.rand() < 0.1 else f"{abs(rng.randn())*100:.2f}"
    r.append(f"{city},{amt},{rng.randint(18,80)},{y}")
CASES["mixed_missing"] = "\n".join(r)

# 23 CRLF line endings
CASES["crlf"] = "\r\n".join(base_rows())

# 24 single column only -> should refuse CLEANLY
CASES["single_column"] = "\n".join(["justone"] + [str(rng.randint(0, 2)) for _ in range(50)])

# 25 empty file -> clean refusal
CASES["empty_file"] = "   \n  \n"

# 26 garbage binary-ish text -> clean refusal or parse
CASES["garbage"] = "\x00\x01\x02 nonsense \x00" * 20

# 27 unnamed columns (leading comma in header)
r = base_rows()
r[0] = ",f1,f2,target"
CASES["unnamed_col"] = "\n".join(r)

# 28 constant + all-empty columns
r = ["f0,const,empty,target"]
for i in range(250):
    y = "1" if rng.rand() < 0.15 else "0"
    r.append(f"{rng.randn()+(1.2 if y=='1' else 0):.3f},7,,{y}")
CASES["constant_empty_cols"] = "\n".join(r)

# ---- byte-level cases (go through any_to_csv like real uploads) ----
BYTE_CASES = {}
_plain = "\n".join(base_rows())

# 29 UTF-16 (Excel "Unicode Text" export)
BYTE_CASES["utf16_file.csv"] = _plain.encode("utf-16")

# 30 Windows-1252 with accents
r = ["city,f0,target"]
for i in range(250):
    y = "1" if rng.rand() < 0.15 else "0"
    r.append(f"Zürich,{rng.randn()+(1.2 if y=='1' else 0):.3f},{y}".encode("cp1252").decode("cp1252"))
BYTE_CASES["latin1_file.csv"] = "\n".join(r).encode("cp1252")

# 31 ZIP archive (Kaggle style)
import zipfile
_b = io.BytesIO()
with zipfile.ZipFile(_b, "w") as z:
    z.writestr("data/train.csv", _plain)
    z.writestr("__MACOSX/junk", "x")
BYTE_CASES["kaggle.zip"] = _b.getvalue()

# 32 GZIP
import gzip
BYTE_CASES["data.csv.gz"] = gzip.compress(_plain.encode())

# 33 JSON records
import json as _json
recs = [{"f0": float(rng.randn() + (1.2 if rng.rand() < 0.15 else 0)),
         "f1": float(rng.randn()), "target": int(rng.rand() < 0.15)} for _ in range(250)]
BYTE_CASES["records.json"] = _json.dumps(recs).encode()

# 34 JSON Lines
BYTE_CASES["records.jsonl"] = "\n".join(_json.dumps(x) for x in recs).encode()

# 35 Stata .dta
_df = pd.DataFrame(recs)
_b = io.BytesIO(); _df.to_stata(_b, write_index=False)
BYTE_CASES["stata.dta"] = _b.getvalue()

# 36 Excel with two junk title rows above the header
_b = io.BytesIO()
_rows = [["Quarterly Report", None, None], ["generated 2026", None, None],
         ["f0", "f1", "target"]]
for i in range(250):
    y = 1 if rng.rand() < 0.15 else 0
    _rows.append([float(rng.randn() + 1.2 * y), float(rng.randn()), y])
pd.DataFrame(_rows).to_excel(_b, index=False, header=False)
BYTE_CASES["excel_title_rows.xlsx"] = _b.getvalue()

# 37 sep= hint line (Excel regional export)
BYTE_CASES["sep_hint.csv"] = ("sep=;\n" + "\n".join(base_rows(sep=";"))).encode()

# 38 parquet magic -> clean refusal
BYTE_CASES["data.parquet"] = b"PAR1" + b"\x00" * 100

# 39 SPSS extension -> clean refusal
BYTE_CASES["data.sav"] = b"$FL2@(#) SPSS DATA FILE" + b"\x00" * 50

BYTE_MUST_REFUSE = {"data.parquet", "data.sav"}

# ---- text-level extras ----
# 40 percent + currency + dates + inf mixed
r = ["signup_date,fee,share,score,target"]
for i in range(260):
    y = "1" if rng.rand() < 0.15 else "0"
    d = f"2024-{rng.randint(1,13):02d}-{rng.randint(1,28):02d}"
    fee = f"${rng.randint(1,9)},{rng.randint(100,999)}.50"
    share = f"{rng.randint(1,99)}%"
    score = "inf" if i == 5 else f"{rng.randn()+(1.2 if y=='1' else 0):.3f}"
    r.append(f'{d},"{fee}",{share},{score},{y}')
CASES["dates_money_percent_inf"] = "\n".join(r)

# 41 multi-line quoted field
r = ["note,f0,target"]
for i in range(200):
    y = "1" if rng.rand() < 0.15 else "0"
    note = '"line one\nline two"' if i == 3 else "plain"
    r.append(f"{note},{rng.randn()+(1.2 if y=='1' else 0):.3f},{y}")
CASES["multiline_quoted"] = "\n".join(r)

MUST_REFUSE = {"single_column", "empty_file", "garbage"}

passed = failed = 0
for name, text in CASES.items():
    try:
        df = engine.read_csv(text)
        p = engine.profile(df)
        r = engine.analyze(text, p["suggested_target"], "clustered_smote",
                           do_classify=True, classifier="erf", folds=1)
        if name in MUST_REFUSE:
            print(f"FAIL {name:24} expected clean refusal, but it processed")
            failed += 1
        else:
            c = r["classify"]
            ok = r["ok"] and c["erf"] is not None
            print(f"{'ok  ' if ok else 'FAIL'} {name:24} target={p['suggested_target']!r:28} "
                  f"warns={len(r['warnings'])}")
            passed += ok; failed += (not ok)
    except ValueError as e:
        if name in MUST_REFUSE:
            print(f"ok   {name:24} clean refusal: {str(e)[:60]}")
            passed += 1
        else:
            print(f"FAIL {name:24} ValueError: {str(e)[:80]}")
            failed += 1
    except Exception as e:
        print(f"FAIL {name:24} {type(e).__name__}: {str(e)[:80]}")
        traceback.print_exc(limit=2)
        failed += 1

for name, blob in BYTE_CASES.items():
    try:
        csv_text, notes = engine.any_to_csv(blob, name)
        df = engine.read_csv(csv_text)
        p = engine.profile(df)
        r = engine.analyze(csv_text, p["suggested_target"], "clustered_smote",
                           do_classify=True, classifier="erf", folds=1)
        if name in BYTE_MUST_REFUSE:
            print(f"FAIL {name:24} expected clean refusal, but it processed")
            failed += 1
        else:
            ok = r["ok"] and r["classify"]["erf"] is not None
            print(f"{'ok  ' if ok else 'FAIL'} {name:24} target={p['suggested_target']!r:24} "
                  f"notes={len(notes)} warns={len(r['warnings'])}")
            passed += ok; failed += (not ok)
    except ValueError as e:
        if name in BYTE_MUST_REFUSE:
            print(f"ok   {name:24} clean refusal: {str(e)[:60]}")
            passed += 1
        else:
            print(f"FAIL {name:24} ValueError: {str(e)[:80]}")
            failed += 1
    except Exception as e:
        print(f"FAIL {name:24} {type(e).__name__}: {str(e)[:80]}")
        traceback.print_exc(limit=2)
        failed += 1

# ---- exact-value regression asserts for the number-parsing / NA / subsample fixes ----
def _clean(cols):
    df = pd.DataFrame(cols); df.attrs["read_warnings"] = []
    return engine._clean_frame(df.copy(), [])

ASSERTS = []
def _chk(name, got, want):
    ok = abs(float(got) - float(want)) < 1e-6
    print(f"{'ok  ' if ok else 'FAIL'} assert:{name:22} got={got} want={want}")
    ASSERTS.append(ok)

_chk("us_thousands",  _clean({"a": ["1,234", "5,678", "9,012", "3,456", "7,890"], "t": [0,1,0,1,0]})["a"][0], 1234)
_chk("eu_decimal",    _clean({"a": ["1,5", "2,7", "3,9", "4,1", "5,3"], "t": [0,1,0,1,0]})["a"][0], 1.5)
_chk("eu_currency",   _clean({"a": ["€1,50", "€2,30", "€9,99", "€4,25", "€7,10"], "t": [0,1,0,1,0]})["a"][0], 1.5)
_chk("us_currency",   _clean({"a": ["$1,234.50", "$2,000.00", "$9,999.99", "$3,000.00", "$7,500.00"], "t": [0,1,0,1,0]})["a"][0], 1234.5)
_chk("percent",       _clean({"a": ["12%", "50%", "7%", "88%", "33%"], "t": [0,1,0,1,0]})["a"][0], 0.12)
# legit '-'/'missing' category labels must survive (not be nulled as NA)
_gr = _clean({"x": list("xyxyx"), "grade": ["A", "-", "B", "missing", "A"], "t": [0,1,0,1,0]})["grade"].tolist()
_chk("na_dash_kept",   1 if "-" in _gr and "missing" in _gr else 0, 1)
# extreme minority: subsample must not collapse it below k-fold viability
_yy = np.array([0]*99950 + [1]*50)
_chk("subsample_floor", int((_yy[engine._proba_subsample(_yy, 1500)] == 1).sum()), 50)

passed += sum(ASSERTS); failed += sum(1 for a in ASSERTS if not a)
total = len(CASES) + len(BYTE_CASES) + len(ASSERTS)
print(f"\n{passed} passed, {failed} failed of {total}")
sys.exit(1 if failed else 0)
