"""Application service layer: thin adapters between the application and the evaluation package.

Nothing here duplicates metric, split, resampling or report logic; the package does that work.
Importing this package does not import pandas, the legacy engine or the live server.

Run the tests from the repository root::

    python -B -m unittest discover -s tests
"""
from .errors import APP_ERROR_CODES, ServiceError, safe_text
from .evaluation import assert_strict_json, evaluate_saved_predictions, render_report_html
from .exports import (FRESH_RUN, PROVENANCE_LABELS, RECORDED_RUN, ExportError, annotate_report, build_repro_bundle,
                      dump_json, effective_settings, replay_argv)
from .profile import profile_for_app
from .recipes import LEGACY_ONLY, RECIPE, legacy_recipe_map, recipe_for_legacy
from .training import app_default_models, train_for_app

__all__ = ["APP_ERROR_CODES", "ExportError", "FRESH_RUN", "LEGACY_ONLY", "PROVENANCE_LABELS", "RECIPE", "RECORDED_RUN",
           "ServiceError", "annotate_report", "app_default_models", "assert_strict_json", "build_repro_bundle",
           "dump_json", "effective_settings", "evaluate_saved_predictions", "legacy_recipe_map", "profile_for_app",
           "recipe_for_legacy", "render_report_html", "replay_argv", "safe_text", "train_for_app"]
