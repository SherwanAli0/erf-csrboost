"""Explicit-label data tools using the shared package, inside the bounded worker.

Author: Sherwan Ali. Row exports require an explicit request; aggregate reports
never include them. The Leakage Lab compares both arms on identical original
held-out rows. Its deliberately contaminated sampler is a teaching intervention.
"""
from __future__ import annotations

import csv
import io
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from rare_event_eval import evaluate_predictions
from rare_event_eval.mixed_models import TabularPreprocessor, make_mixed_model, resample_mixed
from rare_event_eval.splits import make_split
from rare_event_eval.tabular import TabularLimits, parse_table_csv, table_to_dataframe

LIMITS = TabularLimits(max_bytes=2_000_000, max_rows=2000, max_columns=51,
                      max_categorical_cardinality=40, max_expanded_features=200)
OPERATIONS = ('resample', 'score', 'leakage')
METHODS = ('none', 'random_over', 'random_under', 'smote', 'clustered_smote')
MODELS = ('logistic', 'rf_raw', 'erf')


def _frame(text):
    if not isinstance(text, str):
        raise ValueError('Supply a UTF-8 CSV table.')
    return table_to_dataframe(parse_table_csv(text.encode('utf-8'), LIMITS))


def _inputs(body):
    if body.get('confirm_labels') is not True:
        raise ValueError('Confirm the target, exact class labels and feature choices first.')
    frame = _frame(body.get('csv'))
    target, positive, negative = (body.get(k) for k in ('target', 'positive_label', 'negative_label'))
    if not isinstance(target, str) or target not in frame.columns or not isinstance(positive, str) or not isinstance(negative, str) or positive == negative:
        raise ValueError('Choose a target and two different exact class labels.')
    labels = frame[target].to_numpy()
    if set(labels) != {positive, negative}:
        raise ValueError('The target must contain exactly the two declared labels, with no missing labels.')
    excluded = body.get('exclude_columns', [])
    if not isinstance(excluded, list) or any(not isinstance(c, str) or c not in frame.columns or c == target for c in excluded):
        raise ValueError('Excluded columns must be feature columns of the training table.')
    names = [c for c in frame.columns if c != target and c not in excluded]
    if not names or len(names) > 50:
        raise ValueError('Choose between one and 50 predictors.')
    numeric, categorical = [], []
    X = frame[names].to_numpy(dtype=object)
    for i, c in enumerate(names):
        values = frame[c].replace('', np.nan)
        numbers = pd.to_numeric(values, errors='coerce')
        if numbers.notna().sum() == values.notna().sum():
            if np.isinf(numbers.to_numpy(dtype=float)).any():
                raise ValueError('Numeric predictors cannot contain infinite values.')
            X[:, i] = numbers.to_numpy(dtype=float)
            numeric.append(i)
        else:
            if frame[c].nunique() > LIMITS.max_categorical_cardinality:
                raise ValueError('A categorical predictor exceeds the 40-value limit.')
            categorical.append(i)
    y = (labels == positive).astype(int)
    seed = body.get('seed', 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('The seed must be an integer from 0 through 4294967295.')
    columns = dict(numeric_columns=tuple(numeric), categorical_columns=tuple(categorical),
                   column_names=tuple(names), max_expanded_features=200)
    return frame, X, y, target, positive, negative, names, columns, seed


def _csv(frame):
    """Neutralize spreadsheet formulas in string cells/headers only, not numbers."""
    def cell(value):
        if isinstance(value, str) and re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',value):
            return value
        if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@', '\t', '\r')):
            return "'" + value
        return value
    out = io.StringIO(newline='')
    writer = csv.writer(out, lineterminator='\n')
    writer.writerow([cell(c) for c in frame.columns])
    writer.writerows([[cell(v) for v in row] for row in frame.itertuples(index=False, name=None)])
    return out.getvalue()


def _resample(X, y, pre, method, seed):
    if method in ('random_over', 'random_under'):
        from imblearn.over_sampling import RandomOverSampler
        from imblearn.under_sampling import RandomUnderSampler
        sampler = (RandomOverSampler if method == 'random_over' else RandomUnderSampler)(random_state=seed)
        xr, yr = sampler.fit_resample(X, y)
        return xr, yr, {'method': method, 'synthetic_rows': 0,
                        'sample_indices': sampler.sample_indices_.tolist()}
    return resample_mixed(X, y, pre, method=method, retention=.5, seed=seed)


def _decode(coded, pre, names):
    data = {}
    if pre.n_numeric_:
        numbers = pre.scaler_.inverse_transform(coded[:, :pre.n_numeric_])
        for k, pos in enumerate(pre.numeric_columns): data[names[pos]] = numbers[:, k]
    for k, pos in enumerate(pre.categorical_columns):
        vocab = pre.vocabularies_[k]
        data[names[pos]] = [str(vocab[int(v)]) if 0 <= v < len(vocab) else ''
                            for v in coded[:, pre.n_numeric_ + k]]
    return pd.DataFrame(data, columns=names)


def _mapping(original, y, resampled, yr, indices, event):
    """Explicit candidates for duplicate source patterns; never invent source IDs."""
    by_pattern = defaultdict(list)
    for row, label, index in zip(original, y, indices):
        by_pattern[(tuple(row), int(label))].append(int(index))
    synthetic = int(event.get('synthetic_rows', 0))
    rows = []
    for i, (row, label) in enumerate(zip(resampled, yr)):
        is_new = synthetic > 0 and i >= len(resampled)-synthetic
        if is_new: candidates = []
        elif 'sample_indices' in event: candidates = [int(indices[event['sample_indices'][i]])]
        elif event.get('method') in ('none', 'smote') and i < len(indices): candidates = [int(indices[i])]
        else: candidates = by_pattern.get((tuple(row), int(label)), [])
        rows.append({'export_row': i, 'origin': 'synthetic' if is_new else 'original_or_resampled',
                     'source_row_candidates': '|'.join(map(str, candidates[:20])),
                     'candidate_count': len(candidates), 'candidates_truncated': len(candidates)>20})
    return pd.DataFrame(rows)


def execute(body):
    operation = body.get('operation')
    if operation not in OPERATIONS: raise ValueError('Unknown data tool.')
    frame, X, y, target, positive, negative, names, columns, seed = _inputs(body)
    method = body.get('method', 'smote')
    if method not in METHODS: raise ValueError('Unknown resampling method.')
    include_rows = body.get('include_row_data') is True
    out = {'ok': True, 'schema_version': 'fyd-tools-1.0', 'operation': operation,
           'positive_label': positive, 'negative_label': negative, 'target': target,
           'predictors': names, 'seed': seed, 'row_data_included': include_rows,
           'downloads': {}, 'warnings': [
               'Scores are model outputs; calibration is not established.',
               'CSV exports neutralize spreadsheet formula strings; numeric values are not changed.',
               'These tools use the new package recipes, not the legacy or historical research protocol.']}
    if operation == 'score':
        name = body.get('model', 'rf_raw')
        if name not in MODELS: raise ValueError('Unknown scoring model.')
        scoring = _frame(body.get('score_csv'))
        if any(c not in scoring.columns for c in names):
            raise ValueError('The scoring file is missing one or more selected predictors.')
        xs = scoring[names].to_numpy(dtype=object)
        for i in columns['numeric_columns']:
            values = scoring[names[i]].replace('', np.nan)
            numbers = pd.to_numeric(values, errors='coerce')
            if numbers.notna().sum() != values.notna().sum() or np.isinf(numbers.to_numpy(dtype=float)).any():
                raise ValueError('A scoring predictor has invalid numeric values.')
            xs[:, i] = numbers.to_numpy(dtype=float)
        model = make_mixed_model(name, seed=seed, minority_count=int(np.bincount(y).min()), **columns)
        model.fit(X, y)
        p = model.predict_proba(xs)[:, int(np.flatnonzero(model.classes_ == 1)[0])]
        out.update(rows=len(xs), flagged=int((p >= .5).sum()), model=name, threshold=.5,
                   fit_scope='All labelled training rows; no evaluation metrics are claimed for this refit.')
        out['warnings'].append('Do not use known target outcomes or identifiers as predictors. New-row target values, if present, are never used.')
        if include_rows:
            # Never overwrite an uploaded column while adding output scores.
            result = scoring.copy()
            score_name='fyd_positive_score'
            while score_name in result.columns: score_name='_'+score_name
            flag_name='fyd_flag'
            while flag_name in result.columns: flag_name='_'+flag_name
            result[score_name] = p; result[flag_name] = (p >= .5).astype(int)
            out['downloads']['scored_rows.csv'] = _csv(result)
        return out
    split = make_split(y, seed=seed)
    pre = TabularPreprocessor(**columns).fit(X[split.train], y[split.train])
    coded = pre.encode(X)
    xr, yr, event = _resample(coded[split.train], y[split.train], pre, method, seed)
    out.update(partition_counts={key:len(getattr(split,key)) for key in ('train','validation','test')},
               train_counts_before=np.bincount(y[split.train],minlength=2).tolist(),
               train_counts_after=np.bincount(yr,minlength=2).tolist(),
               resampling={k:v for k,v in event.items() if k!='sample_indices'})
    if operation == 'resample':
        out['warnings'].append('The training export contains training-imputed values in original numeric units; categories use the training vocabulary. Validation and test exports use their original cells.')
        if include_rows:
            train = _decode(xr, pre, names); train[target] = [positive if v else negative for v in yr]
            out['downloads']['resampled_train.csv'] = _csv(train)
            out['downloads']['validation.csv'] = _csv(frame.iloc[split.validation])
            out['downloads']['heldout_test.csv'] = _csv(frame.iloc[split.test])
            out['downloads']['row_mapping.csv'] = _csv(_mapping(coded[split.train], y[split.train], xr, yr, split.train, event))
            rows=[{'source_row':int(i),'partition':part} for part in ('train','validation','test') for i in getattr(split,part)]
            out['downloads']['split_indices.csv'] = _csv(pd.DataFrame(rows))
        return out
    if method not in ('smote', 'clustered_smote'):
        raise ValueError('The Leakage Lab requires SMOTE or clustered SMOTE.')
    # Deliberately let the incorrect sampler see ALL labels/features. Only its
    # synthetic suffix is added to the original training rows, not held-out originals.
    all_x, all_y, bad_event = _resample(coded, y, pre, method, seed)
    n = int(bad_event.get('synthetic_rows', 0))
    if not n: raise ValueError('This intervention generated no synthetic rows. Choose an imbalanced table or SMOTE.')
    bad_x=np.concatenate([coded[split.train], all_x[-n:]])
    bad_y=np.concatenate([y[split.train], all_y[-n:]])
    scores={}
    for label, xx, yy in (('train_only_sampler',xr,yr),('contaminated_sampler',bad_x,bad_y)):
        model=RandomForestClassifier(n_estimators=100,random_state=seed,n_jobs=1).fit(pre.expand(xx),yy)
        scores[label]=model.predict_proba(pre.expand(coded[split.test]))[:,1]
    labels=[positive if v else negative for v in y[split.test]]
    report=evaluate_predictions(labels,scores,positive_label=positive,negative_label=negative,
                                score_kind='probability',threshold=.5,review_budgets=[])
    out.update(report=report,evaluation_rows=len(labels),same_original_test_population=True,
               intervention='The incorrect sampler saw validation/test labels and features when creating synthetic training rows. No original validation/test row was directly added to either training arm.')
    out['warnings'] += ['This controlled violation is not an estimate of leakage in your real pipeline. Either arm may score higher.',
                        'The two arms can have different training sizes and synthetic samples. They are not a causal isolation of resampling alone.']
    if include_rows:
        out['downloads']['heldout_scores.csv']=_csv(pd.DataFrame({'source_row':split.test,'label':labels,**scores}))
    return out
