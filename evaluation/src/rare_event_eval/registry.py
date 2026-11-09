"""Stable model registry: identifiers, recipes, parameters and provenance.

Each entry records what ``training.make_model`` actually builds. The recorded
parameters are declarations; ``describe_model`` reads the parameters of the built
object, and the tests compare the two, so a record cannot drift from its
constructor unnoticed.

Reading the table:

* ``params['estimator']`` lists constructor arguments of the final estimator, and
  ``params['resampler']`` (when present) those of a resampling step in front of it.
  The string ``"seed"`` stands for the job seed.
* Every package recipe runs behind the same numeric pipeline: median imputation
  (empty columns kept) and standardization, fitted on the training partition.
* ``supports_mixed`` is true only if the recipe as built accepts categorical input
  without a further adapter. No package recipe does today.
* ``cost_class`` is a qualitative label (light, moderate, heavy). It is not a
  measurement; equal tree counts do not mean equal compute.
* ``scope='package_mixed'`` entries (``MIXED_REGISTRY``, ids ending in ``-mixed``) are the
  mixed-type application variants run by ``mixed.train_comparison_tabular`` when a table has
  categorical predictors. Each has its own ``recipe_id`` and ``supports_mixed=True``, and
  ``variant_of`` names the numeric recipe it adapts. They are not in ``REGISTRY``,
  ``MODEL_NAMES`` or the defaults, and numeric recipe ids and records are unchanged.
* ``scope='legacy_app'`` entries are named so the hosted application can describe
  its older recipes honestly. They are not part of ``MODEL_NAMES`` and are not
  runnable through the training comparison.

This module imports no numerical library.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

SEED = "seed"
PIPELINE_DESCRIPTION = ("SimpleImputer(strategy='median', keep_empty_features=True) then StandardScaler, "
                        "both fitted on the training partition only; numeric inputs only")
SCOPES = ("package", "legacy_app", "package_mixed")
COST_CLASSES = ("light", "moderate", "heavy")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    version: str
    recipe_id: str
    family: str
    params: dict
    supports_mixed: bool
    description: str
    cost_class: str
    scope: str = "package"
    default: bool = False
    variant_of: str | None = None

    def __post_init__(self):
        if self.cost_class not in COST_CLASSES or self.scope not in SCOPES:
            raise ValueError("Invalid cost_class or scope.")

    def resolved_params(self, seed: int = 42) -> dict:
        """Copy of ``params`` with the seed placeholder replaced by ``seed``."""
        def resolve(value):
            if isinstance(value, dict):
                return {key: resolve(item) for key, item in value.items()}
            return seed if value == SEED and isinstance(value, str) else value
        return resolve(self.params)

    def to_record(self, seed: int = 42) -> dict:
        record = asdict(self)
        record["params"] = self.resolved_params(seed)
        record["seed"] = seed
        return record


def _forest(trees, weight, **extra):
    return {"estimator": {"class": "RandomForestClassifier", "n_estimators": trees, "class_weight": weight,
                          "random_state": SEED, "n_jobs": 1, **extra}}


_SPECS = (
    ModelSpec("logistic", "Logistic regression", "1", "rare_event_eval/logistic/1", "linear",
              {"estimator": {"class": "LogisticRegression", "max_iter": 2000, "class_weight": None, "random_state": SEED}},
              False, "Unweighted L2 logistic regression (scikit-learn defaults except max_iter) behind the numeric pipeline.",
              "light", default=True),
    ModelSpec("rf_raw", "Random forest, unweighted", "1", "rare_event_eval/rf_raw/1", "forest",
              _forest(100, None), False,
              "Plain random forest: 100 trees, no class weights, no resampling. This is the unweighted research baseline.",
              "moderate"),
    ModelSpec("rf_weighted", "Random forest, class-weighted", "1", "rare_event_eval/rf_weighted/1", "forest",
              _forest(100, "balanced"), False,
              "Random forest with class_weight='balanced', 100 trees. A different method from the unweighted forest.",
              "moderate", default=True),
    ModelSpec("balanced_rf", "Balanced random forest", "1", "rare_event_eval/balanced_rf/1", "forest",
              {"estimator": {"class": "BalancedRandomForestClassifier", "n_estimators": 100, "sampling_strategy": "all",
                             "replacement": True, "bootstrap": False, "random_state": SEED, "n_jobs": 1}},
              False, "imbalanced-learn BalancedRandomForestClassifier, 100 trees; each tree is fit on a class-balanced sample drawn with replacement (sampling_strategy='all').",
              "moderate", default=True),
    ModelSpec("hist_gradient_boosting", "Histogram gradient boosting", "1", "rare_event_eval/hist_gradient_boosting/1",
              "boosting",
              {"estimator": {"class": "HistGradientBoostingClassifier", "max_iter": 100, "early_stopping": False,
                             "class_weight": None, "random_state": SEED}},
              False, "scikit-learn HistGradientBoostingClassifier, 100 iterations, no early stopping, no class weights.",
              "moderate", default=True),
    ModelSpec("smote_rf", "SMOTE then random forest", "1", "rare_event_eval/smote_rf/1", "resampled_forest",
              {"resampler": {"class": "SMOTE", "k_neighbors": "min(5, training minority count - 1)", "random_state": SEED},
               "estimator": {"class": "RandomForestClassifier", "n_estimators": 100, "class_weight": None,
                             "random_state": SEED, "n_jobs": 1}},
              False, "SMOTE on the preprocessed training partition, then an unweighted 100-tree forest. Numeric features only.",
              "moderate"),
    ModelSpec("csrboost", "CSRBoost-style (application recipe)", "1", "rare_event_eval/csrboost/1", "resampled_boosting",
              {"estimator": {"class": "CSRBoostClassifier", "n_estimators": 50, "retention": 0.5, "random_state": SEED}},
              False,
              "Clustered majority retention (exact KMeans, n_init=10) then SMOTE then AdaBoost over unlimited-depth decision "
              "trees (not stumps). A package recipe; not an exact paper reproduction.",
              "heavy"),
    ModelSpec("erf", "ERF (20 bags x 50 trees)", "1", "rare_event_eval/erf/1", "erf",
              {"estimator": {"class": "ERFClassifier", "n_bags": 20, "trees_per_bag": 50, "retention": 0.5,
                             "resampling": "clustered_smote", "random_state": SEED, "n_jobs": 1}},
              False,
              "Bagged resampled forests, not boosting: 20 bags, each MiniBatchKMeans majority retention plus SMOTE and a "
              "50-tree forest; bag seeds are seed + bag * 7919. Probabilities are averaged over bags.",
              "heavy", default=True),
    ModelSpec("rf_weighted_1000", "Random forest, class-weighted, 1000 trees", "1", "rare_event_eval/rf_weighted_1000/1",
              "forest", _forest(1000, "balanced"), False,
              "Class-weighted forest with 1000 trees, a capacity control for the ERF comparison.", "heavy"),
    ModelSpec("erf_no_clustering", "ERF ablation: SMOTE only", "1", "rare_event_eval/erf_no_clustering/1", "erf",
              {"estimator": {"class": "ERFClassifier", "n_bags": 20, "trees_per_bag": 50, "retention": 0.5,
                             "resampling": "smote", "random_state": SEED, "n_jobs": 1}},
              False, "ERF with clustering removed: 20 bags x 50 trees, SMOTE on all rows of each bag.", "heavy"),
    ModelSpec("erf_no_smote", "ERF ablation: clustering only", "1", "rare_event_eval/erf_no_smote/1", "erf",
              {"estimator": {"class": "ERFClassifier", "n_bags": 20, "trees_per_bag": 50, "retention": 0.5,
                             "resampling": "clustered", "random_state": SEED, "n_jobs": 1}},
              False, "ERF with SMOTE removed: 20 bags x 50 trees, clustered majority retention only.", "heavy"),
    ModelSpec("erf_one_bag", "ERF ablation: one bag", "1", "rare_event_eval/erf_one_bag/1", "erf",
              {"estimator": {"class": "ERFClassifier", "n_bags": 1, "trees_per_bag": 50, "retention": 0.5,
                             "resampling": "clustered_smote", "random_state": SEED, "n_jobs": 1}},
              False, "ERF with a single bag (1 x 50 trees). A capacity ablation, not a faster ERF preset.", "moderate"),
    ModelSpec("legacy_rf_weighted_200", "Legacy FixYourDataset app random forest (200 trees, class-weighted)", "legacy-1",
              "legacy_fyd_app/rf_weighted_200/1", "forest",
              _forest(200, "balanced"), False,
              "This is the legacy FixYourDataset app recipe: RandomForestClassifier, 200 trees, class_weight='balanced' (the app's "
              "make_plain default). It is class-weighted with 200 trees and is NOT the unweighted 100-tree research raw RF. "
              "Listed so the application can name the old recipe honestly; here it uses n_jobs=1 and this package's "
              "numeric pipeline, which differs from the legacy app's preprocessing.",
              "moderate", scope="legacy_app"),
)

REGISTRY: dict[str, ModelSpec] = {spec.id: spec for spec in _SPECS}
if len(REGISTRY) != len(_SPECS):
    raise RuntimeError("Duplicate model id in the registry.")


def _build_mixed_registry() -> dict:
    preprocessing = {"class": "TabularPreprocessor",
                     "numeric": "SimpleImputer(median, keep_empty_features=True) then StandardScaler, fitted on the training partition",
                     "categorical": "training vocabulary, most frequent training value for missing cells, "
                                    "unseen categories get a dedicated code that expands to an all-zero indicator block",
                     "expansion": "one indicator column per vocabulary entry, bounded by the table limits before any model is fitted"}
    resampling = {"class": "SMOTENC", "k_neighbors": "min(5, retained minority count - 1)", "random_state": SEED,
                  "categorical_features": "the integer-coded categorical columns of the coded matrix",
                  "applied": "inside the estimator at fit time, never a pipeline step; needs at least one numeric predictor"}
    cluster_note = (" Clustered retention clusters standardized numeric columns plus unweighted 0/1 indicators of the "
                    "categorical columns (an application representation, not the numeric research recipe).")
    mixed_class = {"smote_rf": "MixedSmoteRFClassifier", "csrboost": "MixedCSRBoostClassifier"}
    entries = {}
    for spec in _SPECS:
        if spec.scope != "package":
            continue
        params = {"preprocessing": preprocessing}
        description = spec.description
        if spec.id in ("logistic", "rf_raw", "rf_weighted", "rf_weighted_1000", "balanced_rf", "hist_gradient_boosting"):
            params["estimator"] = spec.params["estimator"]
            description += " Mixed-type variant: same estimator on the train-fitted mixed preprocessing."
        else:
            estimator = dict(spec.params["estimator"])
            method = estimator.get("resampling")  # ERF variants only
            if spec.id == "smote_rf":
                estimator = {"class": mixed_class["smote_rf"], "n_estimators": 100, "random_state": SEED, "n_jobs": 1}
                description = ("SMOTENC on the coded matrix, then an unweighted 100-tree forest on the expanded representation. "
                               "Mixed-type application recipe; needs at least one numeric predictor.")
                method = "smote"
            elif spec.id == "csrboost":
                estimator["class"] = mixed_class["csrboost"]
                method = "clustered_smote"
                description += " Mixed-type variant: SMOTENC replaces SMOTE." + cluster_note
            else:
                estimator["class"] = "MixedERFClassifier"
                description += " Mixed-type variant."
                if "smote" in method:
                    description += " SMOTENC replaces SMOTE inside each bag."
                if "clustered" in method:
                    description += cluster_note
            params["estimator"] = estimator
            if "smote" in method:
                params["resampling"] = resampling
        mixed_id = spec.id + "-mixed"
        entries[mixed_id] = ModelSpec(
            mixed_id, spec.display_name + ", mixed-type variant", "1", f"rare_event_eval/{spec.id}-mixed/1", spec.family,
            params, True, description, spec.cost_class, scope="package_mixed", variant_of=spec.id)
    return entries


MIXED_REGISTRY: dict[str, ModelSpec] = _build_mixed_registry()


def get_mixed_spec(model_id: str) -> ModelSpec:
    """Mixed-type variant of a package model id (``'erf'`` gives the ``erf-mixed`` entry)."""
    try:
        return MIXED_REGISTRY[model_id + "-mixed"]
    except (KeyError, TypeError):
        raise ValueError("Unknown model name.") from None


def package_model_ids() -> tuple[str, ...]:
    """Ids runnable through the training comparison, in registry order."""
    return tuple(spec.id for spec in REGISTRY.values() if spec.scope == "package")


def default_model_ids() -> tuple[str, ...]:
    return tuple(spec.id for spec in REGISTRY.values() if spec.scope == "package" and spec.default)


def get_spec(model_id: str) -> ModelSpec:
    try:
        return REGISTRY[model_id]
    except (KeyError, TypeError):
        raise ValueError("Unknown model name.") from None


def _plain(value):
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


def describe_model(model_id: str, *, seed: int = 42, minority_count: int = 10) -> dict:
    """Return the registry record and the parameters read from the object ``make_model`` builds.

    ``constructed`` lists every pipeline step with its class and ``get_params(deep=False)``,
    and ``estimator`` / ``resampler`` repeat the final estimator and resampling step.
    Nothing is fitted. ``minority_count`` only affects SMOTE's neighbor count.
    """
    spec = get_spec(model_id)
    from .training import make_model  # lazy: pulls in scikit-learn
    pipeline = make_model(model_id, seed=seed, minority_count=minority_count)
    steps = []
    for name, step in pipeline.steps:
        steps.append({"step": name, "class": type(step).__name__,
                      "params": {key: _plain(value) for key, value in sorted(step.get_params(deep=False).items())}})
    by_name = {step["step"]: step for step in steps}
    result = {"id": spec.id, "spec": spec.to_record(seed), "constructed": steps,
              "estimator": by_name["model"], "resampler": by_name.get("resample")}
    return result


def describe_mixed_model(model_id: str, *, seed: int = 42, minority_count: int = 10) -> dict:
    """Registry record of the mixed variant plus the parameters of the object ``make_mixed_model`` builds.

    Nothing is fitted. ``estimator`` is the final model (a mixed estimator for the resampling
    recipes), ``constructed`` lists every pipeline step.
    """
    spec = get_mixed_spec(model_id)
    from .mixed_models import make_mixed_model  # lazy: pulls in scikit-learn
    pipeline = make_mixed_model(model_id, seed=seed, minority_count=minority_count, numeric_columns=(0,),
                                categorical_columns=(1,), column_names=("numeric", "categorical"))
    steps = [{"step": name, "class": type(step).__name__,
              "params": {key: _plain(value) for key, value in sorted(step.get_params(deep=False).items())}}
             for name, step in pipeline.steps]
    return {"id": spec.id, "spec": spec.to_record(seed), "constructed": steps,
            "estimator": {item["step"]: item for item in steps}["model"]}
