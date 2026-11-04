"""CSRBoost (Yadav et al., IEEE Access 2025) as replicated for this project (a port of the
author's own replication function; not checked against the original authors' code):

  KMeans on the majority (K = n_minority, n_init=10) -> retain 50% per cluster
  -> class-redefinition guard (paper Sec II-D) -> SMOTE (k = min(5, n_min-1))
  -> AdaBoost, T=50, learning_rate=1.0, wrapping a DEFAULT decision tree.

This is the published method, not ERF-CSRBoost."""
import math
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

SEED = 42
T_CSRBOOST = 50
SMOTE_K = 5


def _safe_k(n_min, k=SMOTE_K):
    return max(1, min(k, max(1, n_min - 1)))


def _make_smote(rs, n_min):
    return SMOTE(k_neighbors=_safe_k(n_min), random_state=rs)


def _make_adaboost_csrboost(rs, n_estimators=T_CSRBOOST):
    """CSRBoost (Yadav 2025): AdaBoost wrapping a default decision tree, T=50."""
    base = DecisionTreeClassifier(random_state=rs)
    kwargs = {"estimator": base, "n_estimators": n_estimators,
              "learning_rate": 1.0, "random_state": rs}
    import sklearn
    ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if ver < (1, 6):
        kwargs["algorithm"] = "SAMME"
    if ver < (1, 2):
        kwargs["base_estimator"] = kwargs.pop("estimator")
    return AdaBoostClassifier(**kwargs)


def _fit_resample_safe(sampler, X, y):
    n_min = int(np.sum(y == 1))
    if n_min < 2:
        return X, y
    try:
        Xr, yr = sampler.fit_resample(X, y)
        if len(np.unique(yr)) < 2:
            return X, y
        return Xr, yr
    except Exception:
        return X, y


def csrboost_fit_predict(X_tr, y_tr, X_te, rs=SEED):
    """Returns P(minority) on X_te. Port of the project's CSRBoost replication function."""
    X_min = X_tr[y_tr == 1]
    X_maj = X_tr[y_tr == 0]
    n_min = len(X_min)
    n_maj = len(X_maj)
    K = max(1, min(n_min, n_maj))    # Eq.6 with p=100%, capped at n_majority
    rng = np.random.RandomState(rs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=K, n_init=10, random_state=rs)
        labels = km.fit_predict(X_maj)
    kept = []
    for k in range(K):
        cidx = np.where(labels == k)[0]
        n_keep = max(1, math.ceil(len(cidx) * 0.50))
        if n_keep >= len(cidx):
            kept.extend(cidx.tolist())
        else:
            kept.extend(rng.choice(cidx, n_keep, replace=False).tolist())
    X_ret = np.vstack([X_min, X_maj[kept]])
    y_ret = np.concatenate([np.ones(len(X_min)), np.zeros(len(kept))]).astype(int)
    swapped = False
    if int(np.sum(y_ret == 1)) > int(np.sum(y_ret == 0)):
        y_ret = 1 - y_ret  # class redefinition guard (paper Sec II-D, scoped to SMOTE)
        swapped = True
    Xb, yb = _fit_resample_safe(_make_smote(rs, int(np.sum(y_ret == 1))), X_ret, y_ret)
    clf = _make_adaboost_csrboost(rs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(Xb, yb)
    proba = clf.predict_proba(X_te)[:, 1]
    if swapped:
        proba = 1.0 - proba
    return proba
