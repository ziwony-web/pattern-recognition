# Reproduction

## Inputs

The owner supplied the actual course `train.csv` and `test.csv`; both were inspected and used on 2026-10-14. Keep them locally under `data/`. They are intentionally not distributed in this repository. The old `final code/prediction.csv` is a two-byte placeholder, not a real prediction file.

Train: 22,200 × 49; test: 9,515 × 47. Train labels match `shares > 1400` exactly. Test has neither `y` nor `shares`. The baseline excludes `y`, `shares`, `id`, `url` and `timedelta` from predictors; remaining columns total 46 for these files. Do not substitute the public UCI download and describe it as the same course split.

## Commands

From the repository root, after copying the CSVs into `data/`:

```bash
python -m pip install -r requirements.txt
python scripts/train_baseline.py --train data/train.csv --test data/test.csv --model random_forest
python scripts/train_baseline.py --train data/train.csv --test data/test.csv --model xgboost
```

Omit `--test` to run validation only. Use `--output <folder>` to change output location. Default folders are `results/new-baseline/random_forest/` and `results/new-baseline/xgboost/`.

## Outputs

- `metrics.json`: validation scores, parameters, data hash, validation-row hash and package versions.
- `validation_predictions.csv`: row positions, labels, predictions and probabilities for the held-out partition.
- With `--test`, `test_predictions.csv`: only id, y_predict and y_prob, in original test-row order, from a separate full-data refit.
- With `--test`, `prediction_metadata.json`: input hash and prediction counts; test_metrics is null because no labels exist.

Training-only preprocessing and identical validation rows were checked for both runs. The tested package versions are recorded in [run metadata](../results/verified-2026-10-14/README.md); requirements.txt lists compatible starting bounds rather than a historical environment lock. Results can vary with package versions.

## Historical notebooks

Original notebooks and outputs remain archival. Adjust their Google Drive paths and install Jupyter to inspect them. Their preprocessing differs from the new baseline and some saved cell outputs disagree with cell source. The new script is AI-assisted portfolio maintenance, not code claimed to have been written during the original coursework.
