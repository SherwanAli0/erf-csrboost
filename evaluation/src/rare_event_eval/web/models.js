/* Sherwan Ali. Copy of the runnable package model registry (rare_event_eval.registry, scope package). evaluation/tests/test_web_node.py (ModelListMatchesRegistry)
   fails if this list drifts from the registry: ids, order, display names, recipe ids, families, cost classes, defaults and descriptions. */
(function (root) {
  'use strict';
  const MODELS = Object.freeze([
    {"id":"logistic","name":"Logistic regression","recipe":"rare_event_eval/logistic/1","family":"linear","cost":"light","default":true,"description":"Unweighted L2 logistic regression (scikit-learn defaults except max_iter) behind the numeric pipeline."},
    {"id":"rf_raw","name":"Random forest, unweighted","recipe":"rare_event_eval/rf_raw/1","family":"forest","cost":"moderate","default":false,"description":"Plain random forest: 100 trees, no class weights, no resampling. This is the unweighted research baseline."},
    {"id":"rf_weighted","name":"Random forest, class-weighted","recipe":"rare_event_eval/rf_weighted/1","family":"forest","cost":"moderate","default":true,"description":"Random forest with class_weight='balanced', 100 trees. A different method from the unweighted forest."},
    {"id":"balanced_rf","name":"Balanced random forest","recipe":"rare_event_eval/balanced_rf/1","family":"forest","cost":"moderate","default":true,"description":"imbalanced-learn BalancedRandomForestClassifier, 100 trees; each tree is fit on a class-balanced sample drawn with replacement (sampling_strategy='all')."},
    {"id":"hist_gradient_boosting","name":"Histogram gradient boosting","recipe":"rare_event_eval/hist_gradient_boosting/1","family":"boosting","cost":"moderate","default":true,"description":"scikit-learn HistGradientBoostingClassifier, 100 iterations, no early stopping, no class weights."},
    {"id":"smote_rf","name":"SMOTE then random forest","recipe":"rare_event_eval/smote_rf/1","family":"resampled_forest","cost":"moderate","default":false,"description":"SMOTE on the preprocessed training partition, then an unweighted 100-tree forest. Numeric features only."},
    {"id":"csrboost","name":"CSRBoost-style (application recipe)","recipe":"rare_event_eval/csrboost/1","family":"resampled_boosting","cost":"heavy","default":false,"description":"Clustered majority retention (exact KMeans, n_init=10) then SMOTE then AdaBoost over unlimited-depth decision trees (not stumps). A package recipe; not an exact paper reproduction."},
    {"id":"erf","name":"ERF (20 bags x 50 trees)","recipe":"rare_event_eval/erf/1","family":"erf","cost":"heavy","default":true,"description":"Bagged resampled forests, not boosting: 20 bags, each MiniBatchKMeans majority retention plus SMOTE and a 50-tree forest; bag seeds are seed + bag * 7919. Probabilities are averaged over bags."},
    {"id":"rf_weighted_1000","name":"Random forest, class-weighted, 1000 trees","recipe":"rare_event_eval/rf_weighted_1000/1","family":"forest","cost":"heavy","default":false,"description":"Class-weighted forest with 1000 trees, a capacity control for the ERF comparison."},
    {"id":"erf_no_clustering","name":"ERF ablation: SMOTE only","recipe":"rare_event_eval/erf_no_clustering/1","family":"erf","cost":"heavy","default":false,"description":"ERF with clustering removed: 20 bags x 50 trees, SMOTE on all rows of each bag."},
    {"id":"erf_no_smote","name":"ERF ablation: clustering only","recipe":"rare_event_eval/erf_no_smote/1","family":"erf","cost":"heavy","default":false,"description":"ERF with SMOTE removed: 20 bags x 50 trees, clustered majority retention only."},
    {"id":"erf_one_bag","name":"ERF ablation: one bag","recipe":"rare_event_eval/erf_one_bag/1","family":"erf","cost":"moderate","default":false,"description":"ERF with a single bag (1 x 50 trees). A capacity ablation, not a faster ERF preset."}
  ].map(row => Object.freeze(row)));
  root.FYD = root.FYD || {};
  root.FYD.MODELS = MODELS;
  if (typeof module !== 'undefined' && module.exports) module.exports = MODELS;
})(globalThis);
