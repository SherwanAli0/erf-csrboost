"""The plain baseline: a standard RandomForest with class weights, trained on the
original imbalanced data. The typical one-line approach a practitioner would try first.
If the special methods cannot beat this, they are not worth using."""
from sklearn.ensemble import RandomForestClassifier

SEED = 42


def make_plain(weighted=True, n_estimators=200, seed=SEED):
    return RandomForestClassifier(n_estimators=n_estimators, random_state=seed,
                                  class_weight="balanced" if weighted else None, n_jobs=-1)
