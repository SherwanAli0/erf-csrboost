/* Sherwan Ali. Static wording used by the workbench: provenance labels, glossary, split questions, reason texts,
   privacy statements and the research summary. Numbers in RESEARCH are copied from README.md and
   research/RESEARCH-EVIDENCE.md of this repository; evaluation/tests/test_web_content.js checks that each one
   appears in those files. Plain ASCII only. */
(function (root) {
  'use strict';

  // Only URLs that appear verbatim in the repository's own documents are linked: the audit (README.md) and the November record
  // (research/README.md and docs). The repository root URL is not written out in any README or CITATION line.
  const AUDIT_URL = 'https://github.com/SherwanAli0/csrboost-audit';
  const NOVEMBER_RECORD_URL = 'https://github.com/SherwanAli0/erf-csrboost/tree/main/research/results/november2026_aligned_extension';

  /* The four labels a result may carry. The text itself, not a colour, tells them apart. */
  const PROVENANCE = Object.freeze({
    fictional_fixture: { label: 'fictional fixture', mark: 'F',
      explain: 'Invented cases and invented scores, made to show how the software works. It is not a real-world result and not an ERF result.' },
    recorded_app_run: { label: 'recorded app run', mark: 'R',
      explain: 'A real run of this app on a seeded synthetic table, saved so that it can be read without waiting. It is not a historical ERF research result.' },
    fresh_app_run: { label: 'fresh app run', mark: 'N',
      explain: 'Produced just now by this app from the dataset you uploaded. It is not a historical ERF research result.' },
    historical_research_record: { label: 'historical research record', mark: 'H',
      explain: 'Produced earlier by the research code. It was not produced by this app, and this app does not reproduce it.' }
  });

  const SPLIT_MODES = Object.freeze([
    { id: 'stratified', name: 'Another random observation',
      question: 'Will the model be used on new cases drawn from the same population as these rows?',
      detail: 'Rows are assigned at random. The share of positive cases stays about the same in each partition.', needs: null },
    { id: 'feature_group', name: 'A new exact feature pattern',
      question: 'Will the model meet cases whose exact combination of predictor values never appeared in training?',
      detail: 'Rows with identical predictor values stay together in one partition. This is a stricter question than a random split when values repeat.', needs: null },
    { id: 'group', name: 'A new entity or group',
      question: 'Will the model be used for new customers, patients, machines or other groups that were not in training?',
      detail: 'All rows of a group stay in one partition. This needs a group column.', needs: 'group' },
    { id: 'chronological', name: 'A later time period',
      question: 'Will the model score a later period than the one it learned from?',
      detail: 'Earlier rows train, the next rows validate and the latest rows are tested. This needs a time column that is numeric or a date. A group column cannot be combined with it, and sigmoid calibration is not available with it.', needs: 'time' }
  ]);

  const CALIBRATIONS = Object.freeze([
    { id: 'none', name: 'None', detail: 'Scores are the model output as it comes.' },
    { id: 'sigmoid', name: 'Sigmoid, fitted with training-only cross-validation',
      detail: 'Adds fits and needs enough training cases of each class in every fold. Thresholds are still chosen on the separate validation partition.' }
  ]);

  const GLOSSARY = Object.freeze({
    ap: { term: 'Average precision (AP)',
      text: 'AP summarizes how well the scores put true positives ahead of the other cases. 1 is a perfect ranking. A model with no ranking skill scores about the share of positive cases in the test partition, shown as the baseline. AP does not use a threshold. With few positive cases, AP can change noticeably under a different split, and this report has one split and no uncertainty interval.' },
    roc_auc: { term: 'ROC AUC',
      text: 'The chance that a randomly chosen positive case receives a higher score than a randomly chosen negative case. 0.5 means no ranking skill and 1 is perfect. It does not depend on how rare the positive cases are, so it can look high while most flagged cases are not positive.' },
    threshold: { term: 'Threshold',
      text: 'A threshold turns a score into a decision: cases with a score at or above it are flagged. Here each model gets its own threshold, chosen on the validation partition only, and it is applied unchanged to the test partition.' },
    precision_recall: { term: 'Precision, recall and F1',
      text: 'Precision is the share of flagged cases that are positive. Recall is the share of all positive cases that were flagged. F1 combines them as 2TP / (2TP + FP + FN). A value is unavailable when its denominator is zero.' },
    review_budget: { term: 'Review budget',
      text: 'A review budget flags the top k cases by score, for example the share you can afford to check. It is a count of cases, not a false-positive rate. Tied scores at the cut can change which cases are chosen, and the report records the possible range when that happens.' },
    brier: { term: 'Brier score and log loss',
      text: 'Both compare the predicted probability with the 0 or 1 outcome and are lower when better. Log loss punishes confident wrong probabilities more. Neither one certifies that probabilities are calibrated.' },
    reliability: { term: 'Reliability bins',
      text: 'Cases are grouped by predicted probability. In each group the average score is set beside the share of cases that were positive. The bins are descriptive, and a bin with few cases is noisy.' },
    cost: { term: 'Assumed cost',
      text: 'Costs are the numbers you typed, in one unit of your choice, applied as false positives times the false-positive cost plus false negatives times the false-negative cost plus flagged cases times the review cost. They are assumptions. They are not measured savings.' },
    partitions: { term: 'Train, validation and test',
      text: 'The train partition fits the model. The validation partition selects the threshold. The test partition is not used for any choice and gives the numbers reported. All models in one run share the same three partitions.' },
    overlap: { term: 'Repeated feature patterns',
      text: 'Cases with identical predictor values can end up on both sides of a split. This is a diagnostic. It is not proof of leakage, and choosing a group split changes the question being asked.' },
    cost_class: { term: 'Cost class',
      text: 'light, moderate or heavy is a qualitative label from the model registry. It is not a measurement, and equal tree counts do not mean equal compute. Measured time is shown separately.' }
  });

  /* Plain-language texts for stable codes. A code that is not listed is shown with the service's own message. */
  const REASONS = Object.freeze({
    MODEL_TIMEOUT: 'The model reached its time limit and was stopped. No result was produced for it.',
    JOB_TIMEOUT: 'The whole run reached its time limit and was stopped.',
    QUEUE_WAIT_EXCEEDED: 'The run waited too long in the queue and was removed. Nothing was fitted.',
    CANCELLED: 'The run was cancelled. Nothing from it is kept.',
    SERVICE_STOPPING: 'The service was stopped while the run was queued or running. Start the analysis again, and upload the dataset again if the service restarted.',
    MEMORY_LIMIT_EXCEEDED: 'The run needed more memory than the service allows for one job. Try a smaller file, fewer columns or fewer models.',
    RESULT_TOO_LARGE: 'The report was larger than the service allows. Try fewer models or a smaller file.',
    JOB_CRASHED: 'The worker process ended unexpectedly. Try again, or try fewer models.',
    JOB_INTERNAL_ERROR: 'The service had an internal fault. Try again later.',
    JOB_START_FAILED: 'The service could not start the worker. Try again later.',
    JOB_RESULT_INVALID: 'The worker returned a result the service could not read. Try again.',
    SOME_MODELS_FAILED: 'Some models produced a result and some did not. Each one is listed below with its reason.',
    WORKER_NO_RESULT: 'The model worker ended without a result.',
    MODEL_UNKNOWN: 'The model name is not known to this version.',
    MODEL_DEPENDENCY_MISSING: 'A library this model needs is not installed on the service.',
    CALIBRATION_INFEASIBLE: 'Calibration folds could not be built from the training partition, usually because a class has too few training cases.',
    RESAMPLING_INFEASIBLE: 'Resampling could not run, usually because the training partition has too few positive cases.',
    RESAMPLING_NO_NUMERIC_FEATURES: 'This resampling method needs at least one numeric predictor.',
    RESAMPLING_CLUSTERING_FAILED: 'The clustering step of the resampling method failed.',
    FEATURE_EMPTY_CATEGORY_IN_TRAIN: 'A categorical predictor has no observed value in the training partition.',
    EXPANSION_EXCEEDED: 'The categorical predictors expand to more columns than the limit allows.',
    SPLIT_INFEASIBLE: 'The requested split cannot give both classes in every partition. No other split was substituted.',
    INPUT_INVALID: 'The numeric inputs or labels are not valid for this model.',
    MODEL_OUT_OF_MEMORY: 'The model ran out of memory.',
    MODEL_FIT_ERROR: 'The model could not be fitted. The message below is the library message.',
    LABEL_POSITIVE_ABSENT: 'The positive label you chose does not appear in the target column.',
    LABEL_NEGATIVE_ABSENT: 'The negative label you chose does not appear in the target column.',
    LABEL_MISSING_TARGET: 'Some rows have no target value. Choose to exclude and list them, or fix the file.',
    LABEL_SINGLE_CLASS: 'The target has only one class, so nothing can be compared.',
    LABEL_SINGLETON_CLASS: 'A label occurs only once. It cannot be split across partitions and is not repaired silently.',
    LABEL_MULTICLASS_REQUIRES_OVR: 'The target has more than two labels. Choose one label against all others, or fix the file.',
    TABLE_TOO_MANY_ROWS: 'The table has more rows than the service limit.',
    TABLE_TOO_MANY_COLUMNS: 'The table has more columns than the service limit.',
    TABLE_TOO_LARGE_BYTES: 'The file is larger than the service limit.',
    TABLE_CARDINALITY_EXCEEDED: 'A categorical column has more distinct values than the limit. Exclude it, for example if it is an identifier.',
    TABLE_EXPANSION_EXCEEDED: 'The predictors expand to more columns than the limit. Exclude some columns.'
  });

  const JOB_STATE_TEXT = Object.freeze({
    queued: 'Waiting for the single analysis slot.',
    running: 'Running.',
    succeeded: 'Finished. Every selected model produced a result.',
    partial: 'Finished with a partial result: at least one model produced a result and at least one did not.',
    failed: 'The analysis did not produce a result.',
    cancelled: 'The analysis was cancelled.',
    timed_out: 'The analysis reached a time limit.',
    interrupted: 'The service was stopped before the analysis finished.',
    expired: 'The result was removed from the service after its lifetime. Download what you need before it expires; run the analysis again to see it.'
  });

  const STAGE_TEXT = Object.freeze({
    parsing: 'Reading the uploaded table',
    profiling: 'Checking columns and labels',
    splitting: 'Building the train, validation and test partitions',
    fitting: 'Fitting models',
    evaluating: 'Scoring the test partition',
    reporting: 'Writing the report',
    done: 'Done'
  });

  const HTTP_TEXT = Object.freeze({
    413: 'The file is larger than the service accepts.',
    415: 'The service did not accept that file type for upload.',
    422: 'The service could not use this input.',
    429: 'The service is busy.',
    503: 'The service cannot take this work right now.'
  });

  const EXPORTS = Object.freeze([
    { kind: 'report_html', title: 'Report (HTML)', text: 'One standalone page with the dataset, split, metrics for every model, failures, assumptions and limits. It needs no network to open.' },
    { kind: 'report_json', title: 'Report (JSON)', text: 'The same report as structured data: counts, metrics, thresholds, label mapping, model records and failures.' },
    { kind: 'config_json', title: 'Run settings (JSON)', text: 'The settings of this run, the service limits and the SHA-256 of the uploaded bytes.' },
    { kind: 'repro_bundle', title: 'Reproducibility bundle (ZIP)', text: 'Settings, report, versions and a command for the command-line package, so the run can be repeated on your own copy of the data.' }
  ]);
  const NO_ROW_DATA = 'None of these downloads contains row-level data: no uploaded rows, no predictions and no per-case scores. They do contain aggregate counts, metrics and the names of columns, labels and models, and those names and counts can still be sensitive.';

  const PRIVACY = Object.freeze({
    browser: 'In this mode the file is read by this page in your browser. No request carries the file or its rows to a server, and the page does not keep it in browser storage. Downloads are built in your browser.',
    example: 'The examples are part of this page or load a saved report file. Nothing you own is sent anywhere.',
    hostedBefore: 'Analyzing a dataset here uploads the whole file to the service that serves this page, where models are fitted. The service keeps it only temporarily, and you can delete it after the run. The exact lifetime is shown after the upload. The page does not put the file in browser storage or in the address bar, and it keeps the access tokens for your dataset and run in memory only, so a reload ends access to a run.',
    hostedAfter: (limits) => `Your file is stored by the service for up to ${limits.datasetIdle} without use, and a finished run for ${limits.finishedJob}, then removed. You can remove the upload at any time with the Delete my uploaded data button. The service holds everything in memory and one temporary folder, so a restart removes it too. The page keeps the access tokens for your dataset and run in memory only; a reload ends access to a run.`,
    downloads: NO_ROW_DATA
  });

  /* Numbers below are copied from README.md and research/RESEARCH-EVIDENCE.md. Counts are numbers of datasets,
     not significance tests. */
  const RESEARCH = Object.freeze({
    updated: 'November 2026',
    notPeerReviewed: 'Not peer reviewed.',
    whatIsERF: 'ERF-CSRBoost (ERF) is an experimental extension of CSRBoost (Yadav et al., IEEE Access, 2025). It keeps a CSRBoost-style balancing step and replaces the booster with a bagged ensemble: 20 resampled bags, each a 50-tree random forest, with the probabilities averaged. It is a bagged ensemble, not a boosting method.',
    november: {
      headline: 'Average precision over 20 datasets (15 historical datasets plus five newly aligned datasets; the 20 datasets are 18 distinct feature sets).',
      extension: 'The new extension is 2,000 successful fits: five datasets, four methods and 100 folds each. It is not 2,000 datasets.',
      rows: [
        { against: 'Reimplemented CSRBoost', higher: 16, tied: 0, lower: 4, difference: '+0.0685', note: 'Largest gain +0.3548 (CARGOOD), largest loss -0.0579 (FLAREF).' },
        { against: 'Balanced random forest, 100 trees', higher: 16, tied: 0, lower: 4, difference: '+0.0093', note: '1,000 ERF trees against 100.' },
        { against: 'Plain random forest, 100 trees', higher: 12, tied: 2, lower: 6, difference: '-0.0008', note: 'Mean AP 0.7872 for ERF against 0.7880. The mean difference is slightly negative; it shows neither an advantage nor equivalence.' }
      ],
      limits: [
        'Four datasets (CARGOOD, satimage, thyroid_sick, mammography) supply 65.7% of the summed difference over CSRBoost. On the other 16 the mean difference is about +0.0294. This split was made after inspecting the results, so it is descriptive. A plain random forest gains +0.0693 over CSRBoost, slightly more than ERF.',
        'Model capacity is unequal: ERF has 1,000 trees, the forests 100 trees and CSRBoost 50 stumps. A tree-count-matched comparison exists, but no matched-compute study has been completed. ERF median fit-and-score time is 6.6 to 35.6 times that of a plain random forest on the five new datasets.',
        'The 20 datasets were also used while the method was developed, so they are not an independent test.',
        'An older 18-dataset record uses a different protocol. Against a 1,000-tree balanced forest its mean AP difference was +0.0010 (ERF higher on 12, lower on 6; bootstrap 95% interval -0.0171 to +0.0180), which does not establish an advantage.',
        'A seven-dataset check with a balanced-bagging variant is incomplete: it has six missing metric rows on YEAST5ERL, so it is not used as evidence.'
      ]
    },
    pilot: {
      intro: 'A separate 144-cell controlled pilot used package 0.2.0, approximately 60/20/20 primary stratified partitions and feature-group sensitivity splits. It is a different protocol from the 20-dataset record, and its numbers are not combined with it.',
      points: [
        'ERF had higher mean AP than a balanced random forest on 4 of 5 dataset families (the fifth, WINE, tied).',
        'The predeclared gate for a larger study needed at least 0.01 AP over the balanced forest on 4 families and a median fit-time ratio of at most 3. ERF met the AP condition on 2 of 5 families and the observed ratio was 5.97, so the gate was not met.',
        'A plain random forest had higher mean AP than ERF on 3 of 5 families (SONAR, GLASS and YEAST5). Most of ERF advantage over the balanced forest comes from YEAST5.',
        'Removing the clustering step did not lower mean AP on any of the five families, so the pilot gives no support for that step. Reducing ERF to one bag lowered it on 4 of 5.'
      ],
      caution: 'Two correlated splits per family, five families that had already been inspected and small positive counts. The cell records are private; only aggregates are shown.'
    },
    adaptive: 'A separate adaptive campaign planned 480 cells: 382 newly successful, 96 reused and 2 failed. These are separate procedures and protocols. They are not the headline 20-dataset AP evaluation, and the records are private.',
    audit: {
      text: 'A separate replication audit of published CSRBoost comparisons found 42 of 143 (29.4%) within its 3-point tolerance under one uniform protocol, and 101 stored tuned rows were not regenerated. These statements belong to that audit. They say nothing about ERF or about this app.'
    },
    appVersusResearch: 'This app uses new recipes and a validation-selected threshold on one split of your data. It does not reproduce the research tables. A result produced here is labelled a fresh app run, and it is never presented as a historical research record.',
    reproducibility: 'The saved November tables and their SHA-256 manifest are public and can be checked. The 2,000 fits cannot be rerun from the public repository, because the code that ran them and the prepared arrays are private.'
  });

  const HOW = Object.freeze({
    steps: [
      { title: 'Upload and profile', text: 'The file is read once by the same table readers as the older app. The profile lists columns, types, missing values and a suggested target. Nothing is applied for you: you confirm the target, the positive class and every excluded column.' },
      { title: 'One split, made once', text: 'The rows are split about 60/20/20 into train, validation and test. The split is built once and shared by every model, and the achieved counts are in the report. A split that cannot give both classes in every partition fails with a reason; no other split is substituted.' },
      { title: 'Fit on the train partition only', text: 'Each model runs in its own bounded worker process. Imputation, scaling, category vocabularies, resampling and any calibration are fitted on the train partition (calibration with cross-validation inside it).' },
      { title: 'Choose the threshold on validation data', text: 'For each model, 101 fixed thresholds from 0.00 to 1.00 are tried on the validation partition, maximizing F1 or minimizing assumed cost, and the highest value wins a tie. Test labels are never used for the choice.' },
      { title: 'Report on the untouched test partition', text: 'AP, ROC AUC, the threshold outcome, review budgets and, for probabilities, Brier score and log loss are computed on the test partition. Moving a threshold afterwards would be exploration, not a validated choice.' }
    ],
    leakage: [
      'Fitted statistics, vocabularies and resampling use the train partition only. That is the leakage boundary this app enforces.',
      'The app cannot see what your columns mean. A column that is only known after the outcome, or that encodes it, will make results look better than they will be in use. Exclude such columns, and treat repeated feature patterns across partitions as a diagnostic, not proof of leakage.',
      'The service does not check the data for time order or label quality beyond the profile and the package report.'
    ],
    recipes: [
      'The recipes are the package recipes. The older app used different settings under similar names.',
      'The older app plain forest is a class-weighted 200-tree forest. The package random forest without weights is 100 trees and is the only recipe called plain or unweighted.',
      'ERF is a bag of resampled forests, not boosting. The CSRBoost-style recipe uses exact k-means, SMOTE and AdaBoost over full-depth trees, while the research runner used mini-batch k-means and depth-1 stumps.',
      'None of the CSRBoost versions is the original authors code, and the app ERF is not numerically identical to the research ERF.'
    ],
    failure: [
      'The failure policy is strict. A model that cannot be built or fitted is reported with a reason code and is never replaced by another recipe.',
      'A run in which some models fail is shown as partial, with the failed models listed beside the successful ones.',
      'Limits are provisional presets that have not been measured on a hosted machine: one active analysis, one waiting, a time limit per model and per run, and row, column and memory limits.'
    ],
    notEstablished: [
      'No confidence interval and no significance test is produced, and a model is never named the winner automatically.',
      'Probability metrics and reliability bins do not certify calibration.',
      'Costs are assumptions, not measured savings.',
      'One split gives one estimate. A different split can order close models differently.'
    ]
  });

  const SOURCE_URL = 'https://github.com/SherwanAli0/erf-csrboost';
  const api = { SOURCE_URL, AUDIT_URL, NOVEMBER_RECORD_URL, PROVENANCE, SPLIT_MODES, CALIBRATIONS, GLOSSARY, REASONS, JOB_STATE_TEXT,
    STAGE_TEXT, HTTP_TEXT, EXPORTS, NO_ROW_DATA, PRIVACY, RESEARCH, HOW };
  root.FYD = root.FYD || {};
  root.FYD.content = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
