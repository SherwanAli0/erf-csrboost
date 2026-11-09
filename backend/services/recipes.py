"""Map every classifier, resampling method and model name used by the live app to a registry recipe.

The live app (``backend/engine.py``) compares three fixed models (plain, csrboost, erf) and offers nine
resampling methods and three classifiers for the balanced-data export, the pipeline snippet and
"score new rows". Each name below is mapped to a recipe id of ``rare_event_eval.registry`` or marked
``legacy only`` with the reason. A mapped recipe is the stand-in, not an identical implementation: the
``differences`` field lists what changes. ``docs/RECIPE-DIFFERENCES.md`` explains them in prose.

This module only holds the mapping and imports neither the engine nor the package.
"""
from __future__ import annotations

from copy import deepcopy

RECIPE = "recipe"
LEGACY_ONLY = "legacy only"

_ENTRIES = (
    # --- the three models of the live comparison ------------------------------------------------
    {"legacy_name": "plain", "kind": "model", "used_by": "model comparison, multiclass fallback",
     "legacy_definition": "RandomForestClassifier, 200 trees, class_weight balanced, trained on the original data "
                          "(make_plain with weighted=True); n_jobs=-1",
     "disposition": LEGACY_ONLY, "recipe_id": None, "nearest_recipe": "rf_weighted",
     "reason": "The registry lists this recipe as legacy_rf_weighted_200 (scope legacy_app) so it can be named "
               "honestly, but it is not runnable through the training comparison. rf_raw is the unweighted "
               "100-tree research forest and rf_weighted is the weighted 100-tree forest; neither is this recipe.",
     "differences": ["200 trees versus 100", "class-weighted, so not the unweighted research forest",
                     "legacy preprocessing (median, scaler, most-frequent plus one-hot) versus the package adapter"]},
    {"legacy_name": "csrboost", "kind": "model", "used_by": "model comparison, fold comparison",
     "legacy_definition": "csrboost_exact: exact KMeans (n_init 10) on the majority, keep 50 percent per cluster, "
                          "class-redefinition guard, SMOTE, AdaBoost of 50 full-depth trees; silent unresampled "
                          "fallback when SMOTE cannot run",
     "disposition": RECIPE, "recipe_id": "csrboost", "mixed_recipe_id": "csrboost-mixed", "nearest_recipe": None,
     "reason": "Same clustered retention, SMOTE and AdaBoost over full-depth trees.",
     "differences": ["no silent fallback: an infeasible SMOTE is a failed model with a reason code",
                     "the retained-class guard is replaced by count-based class roles",
                     "resampling runs on the numeric pipeline (standardized), or SMOTENC on a coded matrix when "
                     "categorical predictors are present, instead of SMOTE on one-hot columns",
                     "no 4,000-row training cap"]},
    {"legacy_name": "erf", "kind": "model", "used_by": "model comparison, fold comparison, score new rows",
     "legacy_definition": "20 bags, each MiniBatchKMeans retention (50 percent) plus SMOTE or SMOTE-NC with seed "
                          "42 + bag x 7919, then a 50-tree forest; probabilities averaged; not boosting",
     "disposition": RECIPE, "recipe_id": "erf", "mixed_recipe_id": "erf-mixed", "nearest_recipe": None,
     "reason": "Same bag count, tree count, retention and seed rule.",
     "differences": ["clustering representation for categorical columns (0/1 indicators instead of ordinal codes)",
                     "SMOTENC replaces SMOTE-NC on ordinal codes; synthetic categories stay inside the vocabulary",
                     "no fallback to random oversampling",
                     "no 4,000-row training cap",
                     "n_jobs=1 instead of -1"]},
    # --- classifiers offered for the export, snippet and scoring ---------------------------------
    {"legacy_name": "logreg", "kind": "classifier", "used_by": "balanced-data snippet and score new rows",
     "legacy_definition": "LogisticRegression(max_iter=1000), unweighted unless a weighted baseline is requested",
     "disposition": RECIPE, "recipe_id": "logistic", "mixed_recipe_id": "logistic-mixed", "nearest_recipe": None,
     "reason": "Unweighted logistic regression behind imputation and scaling.",
     "differences": ["max_iter 2000 versus 1000", "no resampling step inside the recipe"]},
    {"legacy_name": "random_forest", "kind": "classifier", "used_by": "balanced-data snippet and score new rows",
     "legacy_definition": "RandomForestClassifier, 200 trees, unweighted unless a weighted baseline is requested",
     "disposition": RECIPE, "recipe_id": "rf_raw", "mixed_recipe_id": "rf_raw-mixed", "nearest_recipe": None,
     "reason": "Unweighted forest; named rf_raw because plain means unweighted.",
     "differences": ["100 trees versus 200", "no resampling step inside the recipe (smote_rf adds SMOTE)"]},
    {"legacy_name": "adaboost", "kind": "classifier", "used_by": "balanced-data snippet and score new rows",
     "legacy_definition": "AdaBoostClassifier with scikit-learn defaults (depth-1 stump base learner, 50 rounds)",
     "disposition": LEGACY_ONLY, "recipe_id": None, "nearest_recipe": None,
     "reason": "The registry has no standalone AdaBoost recipe. AdaBoost appears only inside csrboost, over "
               "full-depth trees after clustered resampling.",
     "differences": []},
    # --- resampling methods -----------------------------------------------------------------------
    {"legacy_name": "clustered_smote", "kind": "resampling_method", "used_by": "export, snippet, model comparison",
     "legacy_definition": "Clustered undersampling of the majority plus SMOTE or SMOTE-NC (the CSRBoost balancing step)",
     "disposition": RECIPE, "recipe_id": "csrboost", "mixed_recipe_id": "csrboost-mixed", "nearest_recipe": None,
     "reason": "Available inside csrboost (exact KMeans) and erf (mini-batch KMeans, 20 bags), not as a standalone "
               "export step.",
     "differences": ["no silent fallback to random oversampling", "see the csrboost and erf entries"]},
    {"legacy_name": "none", "kind": "resampling_method", "used_by": "export, snippet",
     "legacy_definition": "No resampling; the app suggests class weights instead",
     "disposition": RECIPE, "recipe_id": "rf_weighted", "mixed_recipe_id": "rf_weighted-mixed", "nearest_recipe": None,
     "reason": "A model trained on the original data with class weights; rf_raw and logistic are the unweighted options.",
     "differences": ["100 trees versus 200"]},
    {"legacy_name": "smote", "kind": "resampling_method", "used_by": "export, snippet, multiclass fallback",
     "legacy_definition": "SMOTE, or SMOTE-NC on ordinal codes when categorical columns exist; k = min(5, minority - 1); "
                          "falls back to random oversampling on any error, recorded only as a text note",
     "disposition": RECIPE, "recipe_id": "smote_rf", "mixed_recipe_id": "smote_rf-mixed", "nearest_recipe": None,
     "reason": "SMOTE inside the training partition followed by an unweighted 100-tree forest.",
     "differences": ["strict failure (RESAMPLING_INFEASIBLE) instead of the fallback to random oversampling",
                     "SMOTENC on a coded matrix; needs at least one numeric predictor when categories are present",
                     "the resampled training data is not exported by this recipe"]},
    {"legacy_name": "random_over", "kind": "resampling_method", "used_by": "export, snippet",
     "legacy_definition": "RandomOverSampler: copies minority rows; works for any data type",
     "disposition": LEGACY_ONLY, "recipe_id": None, "nearest_recipe": None,
     "reason": "No registry recipe oversamples by copying rows. It remains available in the live app's data export.",
     "differences": []},
    {"legacy_name": "random_under", "kind": "resampling_method", "used_by": "export, snippet",
     "legacy_definition": "RandomUnderSampler: drops majority rows; works for any data type",
     "disposition": LEGACY_ONLY, "recipe_id": None, "nearest_recipe": None,
     "reason": "No registry recipe undersamples at random. It remains available in the live app's data export.",
     "differences": []},
    {"legacy_name": "borderline_smote", "kind": "resampling_method", "used_by": "export, snippet",
     "legacy_definition": "BorderlineSMOTE on numeric data; replaced by random oversampling when categorical columns exist",
     "disposition": LEGACY_ONLY, "recipe_id": None, "nearest_recipe": "smote_rf",
     "reason": "No registry recipe uses Borderline-SMOTE, and the legacy app only supports it for numeric data.",
     "differences": []},
    {"legacy_name": "adasyn", "kind": "resampling_method", "used_by": "export, snippet",
     "legacy_definition": "ADASYN on numeric data; replaced by random oversampling when categorical columns exist",
     "disposition": LEGACY_ONLY, "recipe_id": None, "nearest_recipe": "smote_rf",
     "reason": "No registry recipe uses ADASYN, and the legacy app only supports it for numeric data.",
     "differences": []},
    {"legacy_name": "smote_tomek", "kind": "resampling_method", "used_by": "export, snippet",
     "legacy_definition": "SMOTE followed by Tomek-link cleaning on numeric data; replaced by random oversampling "
                          "when categorical columns exist",
     "disposition": LEGACY_ONLY, "recipe_id": None, "nearest_recipe": "smote_rf",
     "reason": "No registry recipe cleans with Tomek links, and the legacy app only supports it for numeric data.",
     "differences": []},
    {"legacy_name": "cluster_centroids", "kind": "resampling_method", "used_by": "export, snippet",
     "legacy_definition": "ClusterCentroids undersampling on numeric data; replaced by random oversampling when "
                          "categorical columns exist",
     "disposition": LEGACY_ONLY, "recipe_id": None, "nearest_recipe": None,
     "reason": "No registry recipe replaces majority rows by cluster centroids, and the legacy app only supports it "
               "for numeric data.",
     "differences": []},
)


def legacy_recipe_map() -> list:
    """Every legacy name with its registry recipe or ``legacy only`` and the reason (a deep copy)."""
    return deepcopy(list(_ENTRIES))


def recipe_for_legacy(name: str, kind: str | None = None) -> dict | None:
    """Entry for one legacy name; ``kind`` disambiguates a name that exists in two lists."""
    for entry in _ENTRIES:
        if entry["legacy_name"] == name and (kind is None or entry["kind"] == kind):
            return deepcopy(entry)
    return None
