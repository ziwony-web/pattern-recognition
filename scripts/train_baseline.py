"""New portfolio baseline; does not reproduce the historical course scores."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_pipeline(model_name="random_forest"):
    # Fit every learned preprocessing statistic only on the training partition.
    numeric = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True))])
    categorical = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encode', OneHotEncoder(handle_unknown='ignore')),
    ])
    preprocess = ColumnTransformer([
        ('numeric', numeric, make_column_selector(dtype_include=np.number)),
        ('categorical', categorical, make_column_selector(dtype_exclude=np.number)),
    ])
    if model_name == 'xgboost':
        from xgboost import XGBClassifier
        model = XGBClassifier(objective='binary:logistic', gamma=0.3,
            max_depth=5, learning_rate=0.2, min_child_weight=7,
            n_estimators=50, reg_alpha=0, reg_lambda=100,
            random_state=42, n_jobs=2)
    else:
        model = RandomForestClassifier(n_estimators=200, max_depth=22,
            min_samples_split=10, max_features='log2', random_state=42, n_jobs=2)
    return Pipeline([('preprocess', preprocess), ('model', model)])


def load_training_data(path):
    if not path.is_file() or path.stat().st_size < 10:
        raise ValueError('Provide the original course train.csv; repository placeholders contain no data.')
    frame = pd.read_csv(path)
    if 'y' not in frame or not set(frame['y'].dropna().unique()).issubset({0, 1}) or frame['y'].isna().any():
        raise ValueError('Expected a complete binary y column (0/1) from the course dataset.')
    if frame['y'].nunique() != 2 or frame['y'].value_counts().min() < 5:
        raise ValueError('At least five rows per class are required for the stratified holdout.')
    X = frame.drop(columns=['y', 'shares', 'id', 'url', 'timedelta'], errors='ignore')
    if X.shape[1] == 0:
        raise ValueError('No predictor columns remain after excluding target and identifiers.')
    return X, frame['y']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--model', choices=['random_forest', 'xgboost'], default='random_forest')
    parser.add_argument('--test', type=Path, help='Optional unlabeled course test.csv for full-data refit')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path('results/new-baseline') / args.model
    X, y = load_training_data(args.train)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = build_pipeline(args.model)
    pipeline.fit(X_train, y_train)
    probability = pipeline.predict_proba(X_val)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    metrics = {
        'model': args.model,
        'model_parameters': pipeline.named_steps['model'].get_params(),
        'validation_row_sha256': hashlib.sha256(X_val.index.to_numpy(dtype='<i8').tobytes()).hexdigest(),
        'run_type': 'new_baseline_not_historical_reproduction',
        'accuracy': float(accuracy_score(y_val, prediction)),
        'f1': float(f1_score(y_val, prediction)),
        'roc_auc': float(roc_auc_score(y_val, probability)),
        'threshold': 0.5, 'random_state': 42,
        'train_rows': len(X_train), 'validation_rows': len(X_val),
        'features': X.columns.tolist(),
        'data_sha256': hashlib.sha256(args.train.read_bytes()).hexdigest(),
        'versions': {'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__},
    }
    metrics['mean_score'] = float(np.mean([metrics['accuracy'], metrics['f1'], metrics['roc_auc']]))
    if args.model == 'xgboost':
        import xgboost
        metrics['versions']['xgboost'] = xgboost.__version__
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    pd.DataFrame({'source_row': X_val.index, 'y_true': y_val.to_numpy(),
        'y_predict': prediction, 'y_prob': probability}).to_csv(args.output / 'validation_predictions.csv', index=False)
    if args.test is not None:
        test = pd.read_csv(args.test)
        if 'y' in test or 'shares' in test:
            raise ValueError('Expected an unlabeled test CSV without y or shares.')
        missing = sorted(set(X.columns) - set(test.columns))
        if missing or 'id' not in test or test['id'].isna().any() or test['id'].duplicated().any():
            raise ValueError(f'Test requires unique nonmissing id and all predictors; missing={missing}')
        # Validation scores above belong to the holdout model, never this refit.
        full_model = build_pipeline(args.model)
        full_model.fit(X, y)
        probability_test = full_model.predict_proba(test[X.columns])[:, 1]
        pd.DataFrame({'id': test['id'], 'y_predict': (probability_test >= 0.5).astype(int),
            'y_prob': probability_test}).to_csv(args.output / 'test_predictions.csv', index=False)
        prediction_info = {'fit_rows': len(X), 'prediction_rows': len(test),
            'test_sha256': hashlib.sha256(args.test.read_bytes()).hexdigest(),
            'threshold': 0.5, 'test_metrics': None,
            'note': 'Full-data refit; no test labels supplied, so no test performance is reported.'}
        (args.output / 'prediction_metadata.json').write_text(json.dumps(prediction_info, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
