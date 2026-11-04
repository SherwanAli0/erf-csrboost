#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
comparison_rf_csrboost_erf.py  --  Four-Way Comparison
==============================================================================

Compares four methods on all 15 CSRBoost datasets:

  1. CSRBoost      -- Original paper (Yadav et al., IEEE Access 2025)
                      KMeans on majority (K=n_minority) -> retain 50% per cluster
                      -> SMOTE -> AdaBoost (50 stumps)

  2. ERF-CSRBoost  -- Proposed novelty (this work)
                      Same pipeline x20 bags, Random Forest instead of AdaBoost

  3. RF-Raw        -- Breiman (2001) original Random Forest, no imbalance handling.
                      Standard Breiman/randomForest configuration (not all values
                      are explicitly stated in the paper; conventional values used):
                        n_estimators=100, max_features=sqrt(p), fully grown trees
                        (max_depth=None), bootstrap=True, Gini criterion,
                        no class weights, no imbalance preprocessing.
                      Reference: Breiman (2001) "Random Forests",
                                 Machine Learning 45(1), 5-32

  4. RF-BRF        -- Balanced Random Forest, Chen, Liaw & Breiman (2004).
                      The EXACT implementable method from the paper (TR 666, p.3):
                        "Draw a bootstrap sample from the minority class.
                         Randomly draw the SAME NUMBER of cases, with replacement,
                         from the majority class."
                      Per tree: n_minority samples from minority +
                                n_minority samples from majority = 2*n_minority total.
                      Implemented via imbalanced-learn BalancedRandomForestClassifier,
                      which faithfully reproduces this balanced bootstrap procedure.
                      n_estimators=100 and max_features=sqrt(p) are inherited from
                      Breiman (2001) since the 2004 paper does not re-specify them.
                      NOTE: The paper's WRF (Weighted RF) is NOT implemented here
                      because its weights were manually tuned per dataset -- there is
                      no universal formula given, so it cannot be fairly reproduced.
                      Reference: Chen, Liaw, Breiman (2004) "Using Random Forest to
                                 Learn Imbalanced Data", UC Berkeley TR 666

Evaluation : RepeatedStratifiedKFold(n_splits=5, n_repeats=20) = 100 folds
Metrics    : ACC -> AUC -> F1 -> AP -> G-Mean  (CSRBoost paper order)
Datasets   : All 15 from CSRBoost paper (Table 2 order)

Author : Sherwan Ali | Uskudar University | Supervisor: Dr. Gamze Uslu
File   : comparison_rf_csrboost_erf.py  (standalone, do not mix with tests)
==============================================================================
"""

from __future__ import annotations

# NOTE: xlrd >= 2.0.1 is required for the DCCC dataset (.xls format).
# Install it via: pip install xlrd>=2.0.1   (or add to requirements.txt)

import argparse
import math
import warnings
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import scipy.stats as sp_stats
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Suppress imbalanced-learn / sklearn version-compatibility noise.
# BalancedRandomForestClassifier uses joblib workers that call
# sklearn.utils.parallel.delayed outside of a sklearn Parallel context,
# producing a UserWarning flood. The warning is harmless -- it is purely an
# internal compatibility advisory and does not affect correctness or results.
# The global filter is needed because joblib workers run in separate threads
# and do not inherit warnings.catch_warnings() context managers from the main
# thread; only a persistent global filter propagates into those workers.
warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*make it possible to propagate.*",
    category=UserWarning,
)

# ==============================================================================
# GLOBAL CONSTANTS
# ==============================================================================

SEED        = 42
SCRIPT_DIR  = Path(__file__).parent.parent  # after 2026-05-04 reorg, scripts/ -> root
DATA_DIR    = SCRIPT_DIR  # data files now live in numbered folders at root
METRIC_KEYS = ("acc", "auc", "f1", "ap", "gmean")
AUC_TIE_TOL = 0.0  # strict comparison: any positive AUC delta is BETTER, any negative is WORSE

# CSRBoost / ERF-CSRBoost settings (paper defaults)
N_TREES  = 50   # AdaBoost stumps (CSRBoost) / RF trees per bag (ERF)
N_BAGS   = 20   # Number of ensemble bags (ERF-CSRBoost only)

# RF-Raw settings -- Breiman (2001) original configuration
# Reference: Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
RF_RAW_N_ESTIMATORS  = 100       # Breiman used 100 trees in experiments
RF_RAW_MAX_FEATURES  = "sqrt"    # sqrt(p) for classification (Breiman's rule)
RF_RAW_MAX_DEPTH     = None      # Fully grown trees, no pruning
RF_RAW_MIN_SAMPLES_SPLIT = 2     # sklearn default; Breiman grew trees to max
RF_RAW_MIN_SAMPLES_LEAF  = 1     # Grow until leaves are pure
RF_RAW_BOOTSTRAP     = True      # Bootstrap sampling with replacement
RF_RAW_CRITERION     = "gini"    # Gini impurity (CART criterion, Breiman)
RF_RAW_CLASS_WEIGHT  = None      # No imbalance correction -- raw original paper

# RF-BRF settings -- Chen, Liaw & Breiman (2004) Balanced Random Forest
# Procedure from TR 666 page 3:
#   Per tree: sample n_minority WITH REPLACEMENT from minority class,
#             sample n_minority WITH REPLACEMENT from majority class.
#   Tree size: 2 * n_minority samples, perfectly balanced every time.
# n_estimators and max_features are NOT stated in the 2004 paper;
# conventional Breiman (2001) values are used for those.
# imbalanced-learn BalancedRandomForestClassifier parameter mapping:
#   sampling_strategy='all'   -- resample ALL classes per tree, so both minority
#                                and majority are drawn fresh each time.
#                                'auto' would leave the minority untouched (only
#                                undersample the majority), which does not match
#                                the 2004 BRF where a bootstrap sample is drawn
#                                from the minority class as well.
#   replacement=True          -- sample WITH REPLACEMENT, matching TR 666 p.3:
#                                "draw a bootstrap sample from the minority class.
#                                 Randomly draw the same number of cases,
#                                 with replacement, from the majority class."
#   bootstrap=False           -- the internal per-tree balanced sampler already
#                                handles the bootstrap draw; setting bootstrap=False
#                                avoids a second independent bootstrap on top of it,
#                                which is the current imblearn default and intended
#                                behaviour for BRF.
RF_BRF_N_ESTIMATORS       = 100     # conventional (not stated in 2004 paper)
RF_BRF_MAX_FEATURES       = "sqrt"  # conventional (not stated in 2004 paper)
RF_BRF_MAX_DEPTH          = None    # fully grown -- stated in 2004 paper
RF_BRF_CRITERION          = "gini"  # CART -- consistent with Breiman 2001
RF_BRF_SAMPLING_STRATEGY  = "all"   # resample all classes per tree (see note above)
RF_BRF_REPLACEMENT        = True    # with replacement -- TR 666 p.3
RF_BRF_BOOTSTRAP          = False   # balanced sampler handles bootstrap; no double-wrap

METHODS = ["CSRBOOST", "ERF_CSRBOOST", "RF_RAW", "RF_BRF"]

# ==============================================================================
# CSRBoost replication anchor values
# ------------------------------------------------------------------------------
# Source: results/replication_audit_final_20260425.csv (verified 2026-04-25/26
# against the per-dataset replication scripts in 01_PSDAS .. 15_FLARE-F).
#
# These are the CSRBoost numbers the report (Graduation_project_final.pdf,
# Tables 1b-19b) presents as the published-paper replication. They use the
# tuned per-dataset configurations (thresholds, metric protocols) that each
# per-dataset script encodes. They are NOT regenerated here. The comparison
# script substitutes these values in place of running CSRBoost so that the
# CSRBoost row in the four-way table matches the report exactly.
#
# ACC is stored on the 0-1 scale (matching the comparison code's convention
# for the other three methods). The report shows it as a percentage; divide
# by 100.
#
# Wilcoxon vs CSRBoost: each fold's CSRBoost value is filled with the same
# anchor (variance = 0), so the test reduces to "is the other method's median
# different from this anchor?" which is what we want to publish anyway.
# ==============================================================================
CSRBOOST_REPL = {
    "PSDAS":    dict(acc=0.7219, auc=0.6391, f1=0.3970, ap=0.2544, gmean=0.6254),
    "ESR":      dict(acc=0.9110, auc=0.8910, f1=0.7940, ap=0.6630, gmean=0.8900),
    "DCCC":     dict(acc=0.6900, auc=0.6358, f1=0.4349, ap=0.2987, gmean=0.6282),
    "BCW":      dict(acc=0.9294, auc=0.9273, f1=0.9065, ap=0.8539, gmean=0.9267),
    "CB":       dict(acc=0.7654, auc=0.7510, f1=0.6762, ap=0.7179, gmean=0.7164),
    "ESDRP":    dict(acc=0.9628, auc=0.9622, f1=0.9522, ap=0.9233, gmean=0.9620),
    "ILPD":     dict(acc=0.6622, auc=0.6323, f1=0.4869, ap=0.3718, gmean=0.6254),
    "GLASS":    dict(acc=0.9459, auc=0.9170, f1=0.8276, ap=0.7206, gmean=0.9129),
    "SEED":     dict(acc=0.9839, auc=0.9879, f1=0.9647, ap=0.9319, gmean=0.9626),
    "WINE":     dict(acc=0.9978, auc=0.9978, f1=0.9439, ap=0.9121, gmean=0.9599),
    "YEAST5ERL":dict(acc=0.9988, auc=0.6050, f1=0.4700, ap=0.4718, gmean=0.4700),
    "CARGOOD":  dict(acc=0.9579, auc=0.9859, f1=0.8175, ap=0.6758, gmean=0.9778),
    "CARVGOOD": dict(acc=0.9994, auc=0.9836, f1=0.9598, ap=0.9255, gmean=0.9977),
    "YEAST5":   dict(acc=0.9816, auc=0.9312, f1=0.7142, ap=0.5387, gmean=0.8899),
    "FLAREF":   dict(acc=0.9362, auc=0.6339, f1=0.2011, ap=0.1008, gmean=0.4670),
}

# Cargood / Carvgood / Flare-F categorical encodings.
# These match the per-column ordinal maps used by the per-dataset replication
# scripts (13_CARGOOD/cargood_replication.py, 14_CARVGOOD/carvgood_replication.py,
# 15_FLARE-F/flaref_replication.py). The previous loaders used a single global
# encoding dict over all columns combined, which silently mapped the same string
# to different codes depending on column ordering and produced a different X
# matrix than the published replication. Fixed 2026-04-26.
CARGOOD_CAT_MAPS = {
    0: {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},  # Buying
    1: {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},  # Maint
    2: {'2': 0, '3': 1, '4': 2, '5more': 3},         # Doors
    3: {'2': 0, '4': 1, 'more': 2},                  # Persons
    4: {'small': 0, 'med': 1, 'big': 2},             # Lug_boot
    5: {'low': 0, 'med': 1, 'high': 2},              # Safety
}
CARVGOOD_CAT_MAPS = CARGOOD_CAT_MAPS  # same six features

FLAREF_CAT_MAPS = {
    0: {'A': 0, 'H': 1, 'K': 2, 'R': 3, 'S': 4, 'X': 5},  # LargestSpotSize
    1: {'C': 0, 'I': 1, 'O': 2, 'X': 3},                  # SpotDistribution
}

# ==============================================================================
# Section 1  DATASET LOADERS
# (identical to erf_csrboost.py -- standalone copy)
# ==============================================================================

def _to_float(df: pd.DataFrame) -> np.ndarray:
    return df.apply(pd.to_numeric, errors="coerce").fillna(0).values

def _dummies(df: pd.DataFrame) -> pd.DataFrame:
    cats = df.select_dtypes(include="object").columns.tolist()
    return pd.get_dummies(df, columns=cats, drop_first=False) if cats else df

def _label_encode_objects(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        values = sorted(out[col].dropna().astype(str).unique())
        mapping = {value: i for i, value in enumerate(values)}
        out[col] = out[col].astype(str).map(mapping).fillna(0)
    return out

class Bundle:
    def __init__(self, X, y):
        self.X = X.astype(float)
        self.y = y.astype(int)
        self.n_samples      = len(y)
        self.n_features     = X.shape[1]
        counts              = np.bincount(y)
        self.majority_count = int(counts.max())
        self.minority_count = int(counts.min())
        self.ir = self.majority_count / max(self.minority_count, 1)

def _load_psdas():
    p  = DATA_DIR / "01_PSDAS" / "data.csv"
    df = pd.read_csv(p, sep=";")
    # Graduate + Dropout = majority (3630), Enrolled = minority (794), IR=4.57
    y  = (df["Target"] == "Enrolled").astype(int)
    df = df.drop(columns=["Target"])
    df = _dummies(df)
    X  = _to_float(df)
    return Bundle(X, y)

def _load_esr():
    folder = DATA_DIR / "02_ESR"
    files  = list(folder.glob("*.csv"))
    df     = pd.read_csv(files[0])
    drop   = [c for c in df.columns if c.lower().startswith("unnamed")]
    if drop:
        df = df.drop(columns=drop)
    tc = "y" if "y" in df.columns else df.columns[-1]
    y  = (pd.to_numeric(df[tc], errors="coerce").fillna(0).astype(int) == 1).astype(int)
    X  = _to_float(df.drop(columns=[tc]))
    return Bundle(X, y)

def _load_dccc():
    folder = DATA_DIR / "03_DCCC"
    xls    = list(folder.glob("*.xls")) + list(folder.glob("*.xlsx"))
    if xls:
        df = pd.read_excel(xls[0], header=1)
    else:
        df = pd.read_csv(list(folder.glob("*.csv"))[0], header=1)
    df = df.iloc[:, 1:]   # drop ID column
    y  = pd.to_numeric(df.iloc[:, -1], errors="coerce").fillna(0).astype(int).values
    X  = _to_float(df.iloc[:, :-1])
    if y.mean() > 0.5:
        y = 1 - y
    return Bundle(X, y)

def _load_bcw():
    p  = DATA_DIR / "04_BCW" / "wdbc.data"
    df = pd.read_csv(p, header=None)
    y  = (df.iloc[:, 1].values == "M").astype(int)
    X  = df.iloc[:, 2:].values.astype(float)
    return Bundle(X, y)

def _load_esdrp():
    p  = DATA_DIR / "05_ESDRP" / "diabetes_data_upload.csv"
    df = pd.read_csv(p)
    # CSRBoost split: Negative is the minority/positive class (200),
    # Positive is the majority/negative class (320). Encode predictors only
    # after removing the target so no class indicator leaks into X.
    y   = (df["class"].astype(str).str.lower() == "negative").astype(int)
    X   = _to_float(_label_encode_objects(df.drop(columns=["class"])))
    return Bundle(X, y)

def _load_cb():
    p  = DATA_DIR / "06_CB" / "sonar.all-data"
    df = pd.read_csv(p, header=None)
    y  = (df.iloc[:, -1].values == "R").astype(int)   # R=Rock (minority, 97); M=Mine (majority, 111)
    X  = df.iloc[:, :-1].values.astype(float)
    if y.mean() > 0.5:
        y = 1 - y
    return Bundle(X, y)

def _load_glass():
    p  = DATA_DIR / "07_GLASS" / "glass.csv"
    df = pd.read_csv(p)
    y  = (df.iloc[:, -1].values == 7).astype(int)     # Type 7 = minority (IR=5.76)
    X  = df.iloc[:, :-1].values.astype(float)
    return Bundle(X, y)

def _load_ilpd():
    p  = DATA_DIR / "08_ILPD" / "ilpd.csv"
    df = pd.read_csv(p)
    y  = (df["Class"].values == 2).astype(int)         # Class 2 = minority (167)
    X  = _to_float(_label_encode_objects(df.drop(columns=["Class"])))
    return Bundle(X, y)

def _load_seed():
    p  = DATA_DIR / "09_SEED" / "seeds_dataset.txt"
    df = pd.read_csv(p, sep=r"\s+", header=None)
    y  = (df.iloc[:, -1].values == 1).astype(int)
    X  = df.iloc[:, :-1].values.astype(float)
    return Bundle(X, y)

def _load_wine():
    p  = DATA_DIR / "10_WINE" / "wine.data"
    df = pd.read_csv(p, header=None)
    # UCI Wine official CSRBoost split: class 3 is the rarest class (48)
    # and classes 1 + 2 are the majority side (130), giving IR=2.71.
    y  = (df.iloc[:, 0].astype(int).values == 3).astype(int)
    X  = df.iloc[:, 1:].values.astype(float)
    return Bundle(X, y)

def _load_yeast5():
    p     = DATA_DIR / "11_YEAST5" / "yeast5.dat"
    lines = [l for l in p.read_text().splitlines()
             if l.strip() and not l.strip().startswith("@")]
    arr   = np.array([[v.strip() for v in l.split(",")] for l in lines if "," in l])
    y     = (arr[:, -1] == "positive").astype(int)
    X     = arr[:, :-1].astype(float)
    return Bundle(X, y)

def _load_yeast5erl():
    p     = DATA_DIR / "12_YEAST5-ERL" / "yeast5erl.dat"
    lines = [l for l in p.read_text().splitlines()
             if l.strip() and not l.strip().startswith("@")]
    arr   = np.array([[v.strip() for v in l.split(",")] for l in lines if "," in l])
    y     = (arr[:, -1] == "positive").astype(int)
    X     = arr[:, :-1].astype(float)
    return Bundle(X, y)

def _load_cargood():
    # Per-column ordinal encoding matches 13_CARGOOD/cargood_replication.py
    p     = DATA_DIR / "13_CARGOOD" / "car-good.dat"
    lines = [l for l in p.read_text().splitlines()
             if l.strip() and not l.strip().startswith("@")]
    arr   = np.array([[v.strip() for v in l.split(",")] for l in lines if "," in l])
    y     = (arr[:, -1] == "positive").astype(int)
    X     = np.empty((arr.shape[0], 6), dtype=float)
    for col in range(6):
        X[:, col] = [CARGOOD_CAT_MAPS[col][v] for v in arr[:, col]]
    return Bundle(X, y)

def _load_carvgood():
    # Per-column ordinal encoding matches 14_CARVGOOD/carvgood_replication.py
    p     = DATA_DIR / "14_CARVGOOD" / "car-vgood.dat"
    lines = [l for l in p.read_text().splitlines()
             if l.strip() and not l.strip().startswith("@")]
    arr   = np.array([[v.strip() for v in l.split(",")] for l in lines if "," in l])
    y     = (arr[:, -1] == "positive").astype(int)
    X     = np.empty((arr.shape[0], 6), dtype=float)
    for col in range(6):
        X[:, col] = [CARVGOOD_CAT_MAPS[col][v] for v in arr[:, col]]
    return Bundle(X, y)

def _load_flaref():
    # Encoding matches 15_FLARE-F/flaref_replication.py:
    # cols 0-1 are categorical (per-column maps), cols 2-10 are already numeric.
    p     = DATA_DIR / "15_FLARE-F" / "flare-F.dat"
    lines = [l for l in p.read_text().splitlines()
             if l.strip() and not l.strip().startswith("@")]
    arr   = np.array([[v.strip() for v in l.split(",")] for l in lines if "," in l])
    y     = (arr[:, -1] == "positive").astype(int)
    X     = np.empty((arr.shape[0], 11), dtype=float)
    X[:, 0] = [FLAREF_CAT_MAPS[0][v] for v in arr[:, 0]]
    X[:, 1] = [FLAREF_CAT_MAPS[1][v] for v in arr[:, 1]]
    for col in range(2, 11):
        X[:, col] = arr[:, col].astype(float)
    return Bundle(X, y)

# Paper Table 2 order
ALL_LOADERS = {
    "PSDAS":     _load_psdas,
    "ESR":       _load_esr,
    "DCCC":      _load_dccc,
    "BCW":       _load_bcw,
    "CB":        _load_cb,
    "ESDRP":     _load_esdrp,
    "ILPD":      _load_ilpd,
    "GLASS":     _load_glass,
    "SEED":      _load_seed,
    "WINE":      _load_wine,
    "YEAST5ERL": _load_yeast5erl,
    "CARGOOD":   _load_cargood,
    "CARVGOOD":  _load_carvgood,
    "YEAST5":    _load_yeast5,
    "FLAREF":    _load_flaref,
}

DEFAULT_DATASETS = list(ALL_LOADERS.keys())

# ==============================================================================
# Section 2  METRICS
# ==============================================================================

def _gmean(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return float(np.sqrt(sens * spec))

def _metrics(y_prob, y_true) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "acc":   float(accuracy_score(y_true, y_pred)),
        "auc":   float(roc_auc_score(y_true, y_prob)),
        "f1":    float(f1_score(y_true, y_pred, zero_division=0)),
        "ap":    float(average_precision_score(y_true, y_prob)),
        "gmean": _gmean(y_true, y_prob),
    }

# ==============================================================================
# Section 3  BUILDING BLOCKS
# ==============================================================================

def _cluster_k(n_min: int, n_maj: int) -> int:
    """K = n_minority (Eq. 6, p=100%), capped at n_majority."""
    return max(1, min(n_min, n_maj))

def _apply_smote(X, y, rs):
    k = min(5, max(1, int(np.sum(y == 1)) - 1))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SMOTE(k_neighbors=k, random_state=rs).fit_resample(X, y)
    except Exception:
        return X, y

def _cluster_retain(X_min, X_maj, K, pct, rs):
    """CSRBoost's cluster-based undersampling: MiniBatchKMeans (single-init,
    mini-batch approximation of KMeans) -> retain pct% per cluster."""
    rng    = np.random.RandomState(rs)
    km     = MiniBatchKMeans(n_clusters=K, n_init=1, max_iter=100,
                             batch_size=1024, random_state=rs)
    labels = km.fit_predict(X_maj)
    kept   = []
    for k in range(K):
        cidx   = np.where(labels == k)[0]
        n_keep = max(1, math.ceil(len(cidx) * pct))
        kept.extend(
            cidx.tolist() if n_keep >= len(cidx)
            else rng.choice(cidx, n_keep, replace=False).tolist()
        )
    X_ret = np.vstack([X_min, X_maj[kept]])
    y_ret = np.concatenate([np.ones(len(X_min)), np.zeros(len(kept))])
    return X_ret, y_ret

def _build_adaboost(rs, n_est=50):
    """AdaBoost with 50 decision stumps (CSRBoost paper configuration)."""
    base   = DecisionTreeClassifier(max_depth=1, random_state=rs)
    kwargs = {"n_estimators": n_est, "random_state": rs}
    import sklearn
    ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if ver < (1, 6):
        kwargs["algorithm"] = "SAMME"
    if ver >= (1, 2):
        kwargs["estimator"] = base
    else:
        kwargs["base_estimator"] = base
    return AdaBoostClassifier(**kwargs)

def _build_rf_erf(rs, n_est=50):
    """Random Forest used inside ERF-CSRBoost bags (50 trees per bag)."""
    return RandomForestClassifier(
        n_estimators=n_est,
        random_state=rs,
        n_jobs=-1,
    )

def _build_rf_raw(rs):
    """
    Raw Random Forest -- standard Breiman/randomForest configuration.
    No imbalance correction. Used directly on raw (unprocessed) training data.
    Reference: Breiman (2001) "Random Forests", Machine Learning 45(1), 5-32.
    Note: not all values below are explicitly stated in the paper; conventional
    values consistent with Breiman (2001) and the randomForest R package are used.
      - n_estimators : 100  (Breiman used 100 trees in his experiments)
      - max_features : sqrt(p) for classification (conventional for classification)
      - max_depth    : None  (fully grown trees, no pruning -- stated in paper)
      - bootstrap    : True  (bootstrap sampling -- stated in paper)
      - criterion    : gini  (CART criterion -- consistent with paper)
      - class_weight : None  (no imbalance handling in original paper)
    """
    return RandomForestClassifier(
        n_estimators      = RF_RAW_N_ESTIMATORS,
        max_features      = RF_RAW_MAX_FEATURES,
        max_depth         = RF_RAW_MAX_DEPTH,
        min_samples_split = RF_RAW_MIN_SAMPLES_SPLIT,
        min_samples_leaf  = RF_RAW_MIN_SAMPLES_LEAF,
        bootstrap         = RF_RAW_BOOTSTRAP,
        criterion         = RF_RAW_CRITERION,
        class_weight      = RF_RAW_CLASS_WEIGHT,
        random_state      = rs,
        n_jobs            = -1,
    )

def _build_rf_brf(rs):
    """
    Balanced Random Forest -- Chen, Liaw & Breiman (2004) BRF.
    Per tree: sample n_minority WITH REPLACEMENT from each class separately,
    giving 2*n_minority perfectly balanced training samples per tree.
    Implements TR 666 page 3: "Draw a bootstrap sample from the minority class.
    Randomly draw the same number of cases, with replacement, from the majority."
    n_estimators=100 and max_features='sqrt' are conventional values used because
    the 2004 paper does not re-specify them.
    imblearn parameter choices:
      sampling_strategy='all'  -- resample all classes per tree (minority drawn too)
      replacement=True         -- with replacement, matching TR 666 wording
      bootstrap=False          -- balanced sampler handles the bootstrap draw;
                                  no second bootstrap wrap on top
    """
    return BalancedRandomForestClassifier(
        n_estimators      = RF_BRF_N_ESTIMATORS,
        max_features      = RF_BRF_MAX_FEATURES,
        max_depth         = RF_BRF_MAX_DEPTH,
        criterion         = RF_BRF_CRITERION,
        sampling_strategy = RF_BRF_SAMPLING_STRATEGY,
        replacement       = RF_BRF_REPLACEMENT,
        bootstrap         = RF_BRF_BOOTSTRAP,
        random_state      = rs,
        n_jobs            = -1,
    )

# ==============================================================================
# Section 4  METHOD RUNNERS
# ==============================================================================

def run_csrboost(X_tr, y_tr, X_te, y_te, dataset_key: str | None = None) -> dict:
    """
    Live CSRBoost: KMeans (K=n_minority) -> retain 50% per cluster -> SMOTE ->
    AdaBoost (50 stumps, max_depth=1).

    Evaluated under the same uniform protocol as ERF_CSRBOOST / RF_RAW / RF_BRF:
    threshold 0.5 on test-set probabilities, single computation per metric.
    Fold variance is real (not zero), so Wilcoxon vs ERF_CSRBOOST is a real
    paired test.

    Numbers will differ from CSRBOOST_REPL anchors. The anchors reproduce the
    paper's reported values under the paper's per-method tuned protocol; this
    function evaluates CSRBoost under the same clean protocol as the other
    three methods so that the four-way comparison is apples-to-apples.
    """
    X_min  = X_tr[y_tr == 1]
    X_maj  = X_tr[y_tr == 0]
    K      = _cluster_k(len(X_min), len(X_maj))
    Xr, yr = _cluster_retain(X_min, X_maj, K, 0.50, SEED)
    Xb, yb = _apply_smote(Xr, yr, SEED)
    clf    = _build_adaboost(SEED, N_TREES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(Xb, yb)
    prob = clf.predict_proba(X_te)[:, 1]
    return {**_metrics(prob, y_te), "status": "ok"}

def run_erf_csrboost(X_tr, y_tr, X_te, y_te) -> dict:
    """
    ERF-CSRBoost: 20 bags, Random Forest (50 trees per bag).
    Same pipeline as CSRBoost, changed step 4: AdaBoost -> RF + ensemble.
    """
    X_min  = X_tr[y_tr == 1]
    X_maj  = X_tr[y_tr == 0]
    K      = _cluster_k(len(X_min), len(X_maj))
    probas = []
    for b in range(N_BAGS):
        rs     = SEED + b * 7919
        Xr, yr = _cluster_retain(X_min, X_maj, K, 0.50, rs)
        Xb, yb = _apply_smote(Xr, yr, rs)
        clf    = _build_rf_erf(rs, N_TREES)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Xb, yb)
        probas.append(clf.predict_proba(X_te)[:, 1])
    prob = np.mean(probas, axis=0)
    return {**_metrics(prob, y_te), "status": "ok"}

def run_rf_raw(X_tr, y_tr, X_te, y_te) -> dict:
    """
    RF-Raw: Breiman (2001) original Random Forest -- no preprocessing.
    StandardScaler is applied (as in all methods here for fair comparison),
    but no clustering, no undersampling, no SMOTE.
    Breiman's configuration: 100 trees, sqrt(p) features, fully grown, Gini.
    """
    clf = _build_rf_raw(SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    return {**_metrics(prob, y_te), "status": "ok"}

def run_rf_brf(X_tr, y_tr, X_te, y_te) -> dict:
    """
    RF-BRF: Balanced Random Forest, Chen, Liaw & Breiman (2004).
    No external preprocessing. The balanced bootstrap is handled internally
    per tree by BalancedRandomForestClassifier.
    Each tree sees 2*n_minority samples, half from each class.
    """
    clf = _build_rf_brf(SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
    return {**_metrics(prob, y_te), "status": "ok"}

DISPATCH = {
    "CSRBOOST":     run_csrboost,
    "ERF_CSRBOOST": run_erf_csrboost,
    "RF_RAW":       run_rf_raw,
    "RF_BRF":       run_rf_brf,
}

# ==============================================================================
# Section 5  UTILITIES
# ==============================================================================

def _wilcoxon(a, b) -> float:
    a_, b_ = np.array(a), np.array(b)
    if np.all(a_ == b_):
        return 1.0
    try:
        _, p = sp_stats.wilcoxon(a_, b_, alternative="two-sided")
        return float(p)
    except Exception:
        return float("nan")

def _verdict(delta, p) -> str:
    if p >= 0.05:
        return "~EQUAL"
    if delta > 0:
        return "BETTER***" if p < 0.001 else "BETTER**" if p < 0.01 else "BETTER*"
    if delta < 0:
        return "WORSE***"  if p < 0.001 else "WORSE**"  if p < 0.01 else "WORSE*"
    return "~EQUAL"

def _table(headers, rows) -> str:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    sep = lambda ch: "+" + "+".join(ch * (w + 2) for w in widths) + "+"
    fmt = lambda cells: "| " + " | ".join(
        str(cells[i]).ljust(widths[i]) for i in range(len(cells))) + " |"
    lines = [sep("-"), fmt(headers), sep("=")]
    for r in rows:
        lines.append(fmt(r))
    lines.append(sep("-"))
    return "\n".join(lines)

# ==============================================================================
# Section 6  EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(datasets, n_splits, n_repeats, output_root):
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir     = output_root / f"comparison_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    total_folds = n_splits * n_repeats

    W = 110
    print("\n" + "=" * W)
    print("  FOUR-WAY COMPARISON: CSRBoost vs ERF-CSRBoost vs RF-Raw vs RF-BRF")
    print("=" * W)
    print(f"  Methods  :")
    print(f"    CSRBOOST     -- Yadav et al. (IEEE Access 2025): KMeans+50%+SMOTE+AdaBoost")
    print(f"    ERF_CSRBOOST -- This work: same pipeline x{N_BAGS} bags + Random Forest")
    print(f"    RF_RAW       -- Breiman (2001): {RF_RAW_N_ESTIMATORS} trees, sqrt(p), fully grown, no preprocessing")
    print(f"    RF_BRF       -- Chen, Liaw & Breiman (2004) BRF: balanced bootstrap per tree")
    print(f"                    exact TR 666 spec: n_minority samples from each class per tree")
    print(f"  Datasets : {', '.join(datasets)}")
    print(f"  CV       : {n_splits}-fold x {n_repeats} repeats = {total_folds} folds")
    print(f"  Metrics  : ACC -> AUC -> F1 -> AP -> G-Mean")
    print(f"  Output   : {out_dir}")
    print("=" * W)

    all_fold_rows    = []
    per_ds_summaries = []
    phase_start      = perf_counter()

    for ds_name in datasets:
        loader = ALL_LOADERS.get(ds_name)
        if loader is None:
            print(f"\n  SKIP -- {ds_name}: no loader registered")
            continue

        print(f"\n{'-' * W}")
        print(f"  DATASET: {ds_name}")
        try:
            bundle = loader()
        except Exception as e:
            print(f"  SKIP -- load error: {e}")
            continue

        print(f"  n={bundle.n_samples}  features={bundle.n_features}  "
              f"IR={bundle.ir:.2f}  minority={bundle.minority_count}  "
              f"majority={bundle.majority_count}")
        print(f"{'-' * W}")

        X, y     = bundle.X, bundle.y
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
        fold_metrics = {m: [] for m in METHODS}
        ds_start     = perf_counter()

        for fold_i, (tr_idx, te_idx) in enumerate(splitter.split(X, y)):
            scaler     = StandardScaler()
            X_tr       = scaler.fit_transform(X[tr_idx])
            X_te       = scaler.transform(X[te_idx])
            y_tr, y_te = y[tr_idx], y[te_idx]

            for m in METHODS:
                try:
                    result = DISPATCH[m](X_tr, y_tr, X_te, y_te)
                except Exception as e:
                    result = {k: float("nan") for k in METRIC_KEYS}
                    result["status"] = f"ERROR: {e}"
                fold_metrics[m].append(result)
                all_fold_rows.append({
                    "dataset": ds_name, "method": m, "fold": fold_i,
                    **{k: result.get(k, float("nan")) for k in METRIC_KEYS},
                })

            if (fold_i + 1) % 10 == 0 or fold_i == total_folds - 1:
                e   = perf_counter() - ds_start
                eta = (total_folds - fold_i - 1) / ((fold_i + 1) / e) if e > 0 else 0
                print(f"    fold {fold_i+1:3d}/{total_folds}  "
                      f"elapsed={e:.0f}s  ETA={eta:.0f}s", end="\r")
        print()

        # Per-dataset summary dict
        ds_sum = {
            "dataset":  ds_name,
            "n":        bundle.n_samples,
            "features": bundle.n_features,
            "ir":       round(bundle.ir, 2),
            "minority": bundle.minority_count,
            "majority": bundle.majority_count,
        }
        for m in METHODS:
            for k in METRIC_KEYS:
                vals = [f[k] for f in fold_metrics[m] if not np.isnan(f.get(k, float("nan")))]
                ds_sum[f"{m}_{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
                ds_sum[f"{m}_{k}_std"]  = float(np.std(vals))  if vals else float("nan")
        per_ds_summaries.append(ds_sum)

        # Per-dataset result table
        def _mean_auc(m):
            return float(np.nanmean([f["auc"] for f in fold_metrics[m]]))

        csr_auc = _mean_auc("CSRBOOST")
        erf_auc = _mean_auc("ERF_CSRBOOST")
        rfr_auc = _mean_auc("RF_RAW")
        rfb_auc = _mean_auc("RF_BRF")

        # Wilcoxon: ERF vs CSRBoost (primary), others vs CSRBoost
        erf_vs_csr_d = erf_auc - csr_auc
        erf_vs_csr_p = _wilcoxon(
            [f["auc"] for f in fold_metrics["ERF_CSRBOOST"]],
            [f["auc"] for f in fold_metrics["CSRBOOST"]])

        print(f"\n  RESULTS for {ds_name}  (primary metric: AUC):")
        headers = ["Method", "ACC", "AUC", "F1", "AP", "G-Mean", "vs CSRBoost AUC", "p-val", "Verdict"]
        rows = []
        for m in METHODS:
            fm  = fold_metrics[m]
            acc = float(np.nanmean([f["acc"]   for f in fm]))
            auc = float(np.nanmean([f["auc"]   for f in fm]))
            f1  = float(np.nanmean([f["f1"]    for f in fm]))
            ap  = float(np.nanmean([f["ap"]    for f in fm]))
            gm  = float(np.nanmean([f["gmean"] for f in fm]))
            if m == "CSRBOOST":
                rows.append([m, f"{acc:.4f}", f"{auc:.4f}", f"{f1:.4f}",
                             f"{ap:.4f}", f"{gm:.4f}", "--", "--", "anchor"])
            elif m == "ERF_CSRBOOST":
                rows.append([m, f"{acc:.4f}", f"{auc:.4f}", f"{f1:.4f}",
                             f"{ap:.4f}", f"{gm:.4f}",
                             f"{erf_vs_csr_d:+.4f}", f"{erf_vs_csr_p:.4f}",
                             _verdict(erf_vs_csr_d, erf_vs_csr_p)])
            else:  # RF_RAW or RF_BRF -- both compared against CSRBOOST
                d = float(np.nanmean([f["auc"] for f in fold_metrics[m]])) - csr_auc
                p = _wilcoxon(
                    [f["auc"] for f in fold_metrics[m]],
                    [f["auc"] for f in fold_metrics["CSRBOOST"]])
                rows.append([m, f"{acc:.4f}", f"{auc:.4f}", f"{f1:.4f}",
                             f"{ap:.4f}", f"{gm:.4f}",
                             f"{d:+.4f}", f"{p:.4f}",
                             _verdict(d, p)])
        print(_table(headers, rows))

    # Save CSVs
    folds_path   = out_dir / "fold_results.csv"
    summary_path = out_dir / "dataset_summary.csv"
    pd.DataFrame(all_fold_rows).to_csv(folds_path,   index=False)
    pd.DataFrame(per_ds_summaries).to_csv(summary_path, index=False)

    # ==================================================================
    # GLOBAL ANALYSIS
    # ==================================================================
    print("\n" + "=" * W)
    print("  GLOBAL ANALYSIS -- ALL DATASETS")
    print("=" * W)

    method_vals = {m: {k: [] for k in METRIC_KEYS} for m in METHODS}
    for row in all_fold_rows:
        for k in METRIC_KEYS:
            v = row.get(k, float("nan"))
            if not np.isnan(v):
                method_vals[row["method"]][k].append(v)

    # Global mean table
    print("\n  GLOBAL MEAN ACROSS ALL DATASETS:")
    g_headers = ["Method", "ACC", "AUC", "F1", "AP", "G-Mean",
                 "delta AUC vs CSRBoost", "Wilcoxon p", "Verdict"]

    csr_aucs = method_vals["CSRBOOST"]["auc"]
    erf_aucs = method_vals["ERF_CSRBOOST"]["auc"]
    rfr_aucs = method_vals["RF_RAW"]["auc"]
    rfb_aucs = method_vals["RF_BRF"]["auc"]

    g_rows = []
    for m in METHODS:
        mv  = method_vals[m]
        row = [m] + [f"{float(np.mean(mv[k])):.4f}" for k in METRIC_KEYS]
        if m == "CSRBOOST":
            row += ["--", "--", "anchor"]
        elif m == "ERF_CSRBOOST":
            d = float(np.mean(erf_aucs)) - float(np.mean(csr_aucs))
            p = _wilcoxon(erf_aucs, csr_aucs)
            row += [f"{d:+.4f}", f"{p:.6f}", _verdict(d, p)]
        elif m == "RF_RAW":
            d = float(np.mean(rfr_aucs)) - float(np.mean(csr_aucs))
            p = _wilcoxon(rfr_aucs, csr_aucs)
            row += [f"{d:+.4f}", f"{p:.6f}", _verdict(d, p)]
        else:  # RF_BRF
            d = float(np.mean(rfb_aucs)) - float(np.mean(csr_aucs))
            p = _wilcoxon(rfb_aucs, csr_aucs)
            row += [f"{d:+.4f}", f"{p:.6f}", _verdict(d, p)]
        g_rows.append(row)
    print(_table(g_headers, g_rows))

    # Per-dataset winner table
    print("\n  PER-DATASET WINNER (AUC -- primary metric):")
    print("  Strict comparison: a method is the winner if its AUC is strictly greater than")
    print("  every other method's AUC. Ties are only declared when AUCs are exactly equal.")
    ds_present = list(dict.fromkeys(r["dataset"] for r in all_fold_rows))
    ds_m_aucs  = {ds: {m: [] for m in METHODS} for ds in ds_present}
    for row in all_fold_rows:
        m = row["method"]
        if m in ds_m_aucs[row["dataset"]]:
            ds_m_aucs[row["dataset"]][m].append(row.get("auc", float("nan")))

    win_headers = ["Dataset", "IR", "n_min",
                   "CSRBoost", "ERF_CSRBoost", "RF_Raw", "RF_BRF",
                   "Winner", "ERF vs CSR", "ERF vs BRF"]
    win_rows = []
    exclusive_wins = {m: 0 for m in METHODS}
    tie_memberships = {m: 0 for m in METHODS}
    tied_datasets = 0

    for ds in ds_present:
        ds_info = next((r for r in per_ds_summaries if r["dataset"] == ds), {})
        aucs = {
            "CSRBOOST":     float(np.nanmean(ds_m_aucs[ds]["CSRBOOST"])),
            "ERF_CSRBOOST": float(np.nanmean(ds_m_aucs[ds]["ERF_CSRBOOST"])),
            "RF_RAW":       float(np.nanmean(ds_m_aucs[ds]["RF_RAW"])),
            "RF_BRF":       float(np.nanmean(ds_m_aucs[ds]["RF_BRF"])),
        }
        c_a, e_a, r_a, b_a = (aucs[m] for m in METHODS)
        best = max(aucs.values())
        leaders = [m for m in METHODS if aucs[m] >= best]
        if len(leaders) == 1:
            winner = leaders[0]
            exclusive_wins[winner] += 1
        else:
            winner = "TIE: " + "/".join(leaders)
            tied_datasets += 1
            for m in leaders:
                tie_memberships[m] += 1
        win_rows.append([
            ds,
            ds_info.get("ir", "?"),
            ds_info.get("minority", "?"),
            f"{c_a:.4f}",
            f"{e_a:.4f}",
            f"{r_a:.4f}",
            f"{b_a:.4f}",
            winner,
            f"{e_a - c_a:+.4f}",
            f"{e_a - b_a:+.4f}",
        ])
    print(_table(win_headers, win_rows))

    print(f"\n  EXCLUSIVE WIN COUNT (by AUC, strict comparison, no tie tolerance):")
    for m, w in exclusive_wins.items():
        print(f"    {m:<16} : {w}/{len(ds_present)} datasets")
    print(f"    {'TIED DATASETS':<16} : {tied_datasets}/{len(ds_present)} datasets")

    print(f"\n  TIE MEMBERSHIP COUNT (datasets where method shares best AUC):")
    for m, w in tie_memberships.items():
        print(f"    {m:<16} : {w}/{len(ds_present)} datasets")

    # Contextual note on the RF variants
    print(f"\n  NOTES ON RF VARIANTS:")
    print(f"  RF-Raw (2001) : No imbalance handling. Bootstrap under-represents")
    print(f"                  minority; majority voting suppresses it further.")
    print(f"  RF-BRF (2004) : Data-level fix INSIDE each tree -- balanced bootstrap")
    print(f"                  (n_minority from each class per tree). Stronger than")
    print(f"                  class weights alone, but still no pre-balancing of the")
    print(f"                  full dataset or cross-bag diversity.")
    print(f"  ERF-CSRBoost  : Full data-level fix + ensemble. CSRBoost pipeline")
    print(f"                  (clustering + undersampling + SMOTE) completely balances")
    print(f"                  the data BEFORE RF sees it, repeated across 20 independent")
    print(f"                  bags with different random seeds. RF works in its ideal")
    print(f"                  setting every single time.")

    elapsed = perf_counter() - phase_start
    print(f"\n  Total elapsed : {elapsed:.0f}s ({elapsed / 3600:.1f}h)")
    print(f"  Results saved : {out_dir}")
    print("=" * W)
    print("  COMPARISON COMPLETE")
    print("=" * W)

# ==============================================================================
# Section 7  ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Four-way comparison: CSRBoost vs ERF-CSRBoost vs RF-Raw vs RF-BRF")
    parser.add_argument(
        "--datasets", nargs="+", default=DEFAULT_DATASETS,
        help="Dataset keys to run (default: all 15)")
    parser.add_argument(
        "--splits",  type=int, default=5,
        help="Number of CV folds (default: 5)")
    parser.add_argument(
        "--repeats", type=int, default=20,
        help="Number of CV repeats (default: 20 => 100 folds total)")
    args = parser.parse_args()

    out_root = SCRIPT_DIR / "outputs"
    out_root.mkdir(exist_ok=True)

    run_experiment(
        datasets    = args.datasets,
        n_splits    = args.splits,
        n_repeats   = args.repeats,
        output_root = out_root,
    )

if __name__ == "__main__":
    main()
