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


def build_pipeline():
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
    return Pipeline([
        ('preprocess', preprocess),
        ('model', RandomForestClassifier(n_estimators=200, max_depth=22,
            min_samples_split=10, max_features='log2', random_state=42, n_jobs=-1)),
    ])


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
    parser.add_argument('--output', type=Path, default=Path('results/new-baseline'))
    args = parser.parse_args()
    X, y = load_training_data(args.train)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    probability = pipeline.predict_proba(X_val)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    metrics = {
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
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    pd.DataFrame({'source_row': X_val.index, 'y_true': y_val.to_numpy(),
        'y_predict': prediction, 'y_prob': probability}).to_csv(args.output / 'validation_predictions.csv', index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
