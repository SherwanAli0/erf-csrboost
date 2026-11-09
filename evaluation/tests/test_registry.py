import unittest

from rare_event_eval.registry import REGISTRY, SEED, ModelSpec, default_model_ids, describe_model, package_model_ids
from rare_event_eval.training import DEFAULT_MODELS, MODEL_NAMES, make_model

ORIGINAL_MODEL_NAMES = ("logistic", "rf_raw", "rf_weighted", "balanced_rf", "hist_gradient_boosting",
                        "smote_rf", "csrboost", "erf", "rf_weighted_1000", "erf_no_clustering",
                        "erf_no_smote", "erf_one_bag")
ORIGINAL_DEFAULTS = ("logistic", "rf_weighted", "balanced_rf", "hist_gradient_boosting", "erf")


class RegistryIdentity(unittest.TestCase):
    def test_ids_and_default_order_are_unchanged(self):
        self.assertEqual(MODEL_NAMES, ORIGINAL_MODEL_NAMES)
        self.assertEqual(DEFAULT_MODELS, ORIGINAL_DEFAULTS)
        self.assertEqual(MODEL_NAMES, package_model_ids())
        self.assertEqual(DEFAULT_MODELS, default_model_ids())

    def test_every_spec_has_complete_provenance(self):
        recipes = set()
        for spec in REGISTRY.values():
            with self.subTest(model=spec.id):
                self.assertIsInstance(spec, ModelSpec)
                for field in ("id", "display_name", "version", "recipe_id", "family", "description"):
                    self.assertTrue(getattr(spec, field))
                self.assertIsInstance(spec.params, dict)
                self.assertIn("estimator", spec.params)
                self.assertIsInstance(spec.supports_mixed, bool)
                self.assertIn(spec.cost_class, ("light", "moderate", "heavy"))
                recipes.add(spec.recipe_id)
        self.assertEqual(len(recipes), len(REGISTRY))

    def test_no_package_recipe_claims_mixed_type_support_yet(self):
        self.assertFalse(any(spec.supports_mixed for spec in REGISTRY.values()))

    def test_unknown_id_is_rejected(self):
        with self.assertRaises(ValueError):
            describe_model("not_a_model")


class RecordsMatchConstructors(unittest.TestCase):
    def test_declared_parameters_equal_the_built_objects(self):
        for model_id, spec in REGISTRY.items():
            with self.subTest(model=model_id):
                for seed in (42, 7):
                    described = describe_model(model_id, seed=seed)
                    declared = spec.resolved_params(seed)
                    self.assertEqual(described["spec"]["params"], declared)
                    for block, built in (("estimator", described["estimator"]), ("resampler", described["resampler"])):
                        if block not in declared:
                            self.assertIsNone(built)
                            continue
                        self.assertEqual(declared[block]["class"], built["class"])
                        for key, value in declared[block].items():
                            if key in ("class", "k_neighbors"):
                                continue
                            self.assertEqual(built["params"][key], value, f"{model_id}.{block}.{key}")

    def test_smote_neighbors_follow_the_training_minority_count(self):
        for minority, expected in ((2, 1), (4, 3), (6, 5), (50, 5)):
            actual = describe_model("smote_rf", minority_count=minority)["resampler"]["params"]["k_neighbors"]
            self.assertEqual(actual, expected)

    def test_every_pipeline_starts_with_the_same_numeric_preprocessing(self):
        for model_id in REGISTRY:
            steps = describe_model(model_id)["constructed"]
            self.assertEqual([step["class"] for step in steps[:2]], ["SimpleImputer", "StandardScaler"], model_id)
            self.assertEqual(steps[0]["params"]["strategy"], "median")

    def test_seed_placeholder_is_replaced_everywhere(self):
        for spec in REGISTRY.values():
            resolved = spec.resolved_params(42)
            self.assertEqual(resolved["estimator"]["random_state"], 42)
            self.assertNotIn(SEED, [value for block in resolved.values() for value in block.values()])


class NamedRecipes(unittest.TestCase):
    def estimator(self, model_id):
        return make_model(model_id).named_steps["model"]

    def test_rf_raw_is_unweighted_and_rf_weighted_is_not(self):
        raw, weighted = self.estimator("rf_raw"), self.estimator("rf_weighted")
        self.assertIsNone(raw.class_weight)
        self.assertEqual(weighted.class_weight, "balanced")
        self.assertEqual((raw.n_estimators, weighted.n_estimators), (100, 100))
        self.assertEqual(self.estimator("rf_weighted_1000").n_estimators, 1000)
        self.assertIsNone(REGISTRY["rf_raw"].params["estimator"]["class_weight"])

    def test_logistic_regression_is_unweighted(self):
        self.assertIsNone(self.estimator("logistic").class_weight)

    def test_erf_is_twenty_bags_of_fifty_trees_and_ablations_differ_as_named(self):
        erf = self.estimator("erf")
        self.assertEqual((erf.n_bags, erf.trees_per_bag, erf.resampling), (20, 50, "clustered_smote"))
        self.assertEqual(self.estimator("erf_one_bag").n_bags, 1)
        self.assertEqual(self.estimator("erf_no_clustering").resampling, "smote")
        self.assertEqual(self.estimator("erf_no_smote").resampling, "clustered")
        self.assertEqual(REGISTRY["erf"].family, "erf")

    def test_legacy_recipe_is_distinct_from_the_research_raw_forest(self):
        legacy, raw = REGISTRY["legacy_rf_weighted_200"], REGISTRY["rf_raw"]
        self.assertEqual(legacy.scope, "legacy_app")
        self.assertNotIn(legacy.id, MODEL_NAMES)
        self.assertNotEqual(legacy.recipe_id, raw.recipe_id)
        self.assertIn("legacy FixYourDataset app recipe", legacy.description)
        built = self.estimator("legacy_rf_weighted_200")
        self.assertEqual((built.n_estimators, built.class_weight), (200, "balanced"))
        self.assertNotEqual((built.n_estimators, built.class_weight), (100, None))

    def test_no_faster_erf_preset_exists_without_its_own_recipe(self):
        erf_recipes = {spec.recipe_id: spec.params["estimator"] for spec in REGISTRY.values() if spec.family == "erf"}
        for recipe, params in erf_recipes.items():
            if recipe != "rare_event_eval/erf_one_bag/1":
                self.assertEqual((params["n_bags"], params["trees_per_bag"]), (20, 50), recipe)
        self.assertEqual(len(erf_recipes), 4)

    def test_registry_module_imports_no_numerical_library(self):
        import rare_event_eval.registry as registry
        with open(registry.__file__, encoding="utf-8") as stream:
            top_level = [line for line in stream.read().splitlines() if line.startswith(("import ", "from "))]
        self.assertTrue(all(not any(name in line for name in ("sklearn", "numpy", "pandas", "imblearn"))
                            for line in top_level))


if __name__ == "__main__":
    unittest.main()
