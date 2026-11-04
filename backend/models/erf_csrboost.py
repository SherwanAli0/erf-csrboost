"""ERF-CSRBoost (Sherwan Ali): an experimental, unpublished extension of CSRBoost.

It is a bagged ensemble of resampled random forests. It is NOT a boosting method and has no
gradient boosting in it.

N_BAGS bags; each bag = CSRBoost-style balancing (clustered undersampling + SMOTE, bag-specific
seed) then a RandomForest (N_TREES trees); the bags' probabilities are averaged.

This is adapted from the research runner (comparison_rf_csrboost_erf.py), not numerically
identical to it. It keeps the bag and tree counts (20 bags x 50 trees), the per-bag seed rule
(SEED + bag * 7919) and the probability averaging. It differs in preprocessing (imputation,
one-hot columns, SMOTE-NC for mixed data), in the scaling used for SMOTE, in the fallback when
SMOTE fails, and in the training-size cap applied by the app. Results from the research runs
are not reproduced by this app."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SkPipeline

SEED = 42
N_BAGS = 20
N_TREES = 50


class ERFModel:
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_erf(X_train, y_train, num, cat, resample_fn, preprocessor_fn):
    """resample_fn(method, X, y, num, cat, seed) and preprocessor_fn(num, cat) are injected
    from the engine so the balancing stays identical to the downloadable output."""
    models = []
    for b in range(N_BAGS):
        s = SEED + b * 7919
        Xb, yb, _ = resample_fn("clustered_smote", X_train, y_train, num, cat, seed=s)
        pipe = SkPipeline([("pre", preprocessor_fn(num, cat)),
                           ("clf", RandomForestClassifier(n_estimators=N_TREES,
                                                          random_state=s, n_jobs=-1))])
        pipe.fit(Xb, yb)
        models.append(pipe)
    return ERFModel(models)
