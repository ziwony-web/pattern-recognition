# Online News Popularity Prediction

**Pattern Recognition · undergraduate team project · tabular binary classification**

Predicting article popularity from content metadata, sentiment, keyword statistics, and topic features. The course dataset is derived from UCI Online News Popularity; the supplied course CSV confirms `y = 1` exactly when `shares > 1,400` (zero mismatches across 22,200 rows). The supplied `y` labels are used directly in the code.

**Start here:** [Project report](docs/report.pdf) · [Final course notebook](final%20code/code.ipynb) · [Experiment evidence](docs/experiments.md) · [Reproduction guide](docs/reproduction.md) · [한국어 이해 가이드](docs/project-guide-ko.md)

## Project at a glance

| Item | Description |
|---|---|
| Task | Predict a binary popularity label and class probability |
| Course training data | 22,200 rows × 49 columns; 46 predictors after excluding id, shares and y |
| Models explored | Random Forest and XGBoost |
| Evaluation | Accuracy, F1, ROC AUC; course score is their arithmetic mean |
| Team | Yunwoo Choi, Seojeong Cho, Ingyeong Yang, Jiwon Kang |
| Current reproducibility | Supplied course CSVs verified; two new baseline runs completed on 2026-10-14; raw CSVs are not distributed here |
| Course test data | 9,515 rows × 47 columns; no ground-truth labels supplied |

## Historical experiments

These are **historical notebook outputs**, not newly reproduced scores or a controlled head-to-head comparison. Different preprocessing and split configurations were used.

| Experiment | Accuracy | F1 | ROC AUC | Mean | Threshold |
|---|---:|---:|---:|---:|---:|
| Random Forest, saved validation run | 0.6788 | 0.6846 | 0.7520 | 0.7051 | 0.48 |
| XGBoost, final course notebook | 0.6587 | 0.7001 | 0.7243 | 0.6944 | 0.414 |

The Random Forest notebook uses an 80/20 split with seed 42 after preprocessing; the final XGBoost notebook uses a 75/25 split with seed 100. These numbers alone do not establish which model generalizes better. [Read the experiment conditions and limitations](docs/experiments.md).

## Approach

1. Inspect missing values and feature groups.
2. Apply feature-dependent imputation and encode channel/weekday categories.
3. Explore tree ensembles with cross-validation and threshold changes.
4. Inspect feature importance and discuss limitations.

The final course pipeline also clips numerical values at the 1st/99th percentiles. Some historical preprocessing is fitted before the validation split; this needs correction in a rerun.

## Jiwon Kang's contribution

Contributed to missing-value treatment and categorical encoding, analysis of selected features, performance comparison, and preparation of explanations for the team report. Team-level model results are reported above; ownership of every modeling experiment is not attributed to one member.

## Code and materials

| Path | Purpose |
|---|---|
| `final code/code.ipynb` | Original final submission workflow and saved outputs |
| `results/Random_Forest_Classfier.ipynb` | Historical Random Forest experiments |
| `results/XGBoost.ipynb` | Historical XGBoost experiments |
| `scripts/train_baseline.py` | New baseline with training-only preprocessing and explicit run metadata |
| `docs/report.pdf` | Original course report |
| `results/figures/` | Existing feature-importance figures |

### Verified maintenance baselines (2026-10-14)

These are **new experiments**, separate from the original coursework. Both use the same stratified 80/20 split (17,760 training / 4,440 validation rows), seed 42, training-fitted median/constant imputation and one-hot encoding, and threshold 0.5. Fixed model configurations were adapted from archival settings; no new grid search or threshold optimization was performed.

| Model | Accuracy | F1 | ROC AUC | Mean |
|---|---:|---:|---:|---:|
| Random Forest | 0.6653 | 0.6685 | 0.7239 | 0.6859 |
| Xgboost | 0.6671 | 0.6664 | 0.7243 | 0.6859 |

The AUC difference is approximately 0.00042 on one split; it does not establish a reliable winner. This split may overlap earlier exploratory validation data and is not a new untouched benchmark. [Machine-readable run evidence](results/verified-2026-10-14/README.md).

Place your supplied CSVs in `data/`, then run from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/train_baseline.py --train data/train.csv --test data/test.csv --model random_forest
python scripts/train_baseline.py --train data/train.csv --test data/test.csv --model xgboost
```

Outputs go to `results/new-baseline/<model>/`. With `--test`, the script additionally refits on all 22,200 labeled rows and writes 9,515 `id,y_predict,y_prob` predictions. Test labels were not supplied, so **no test score is reported**. Raw CSVs and row-level predictions are not committed; the recorded metrics, data hashes, split hash and versions are committed.

## Interpretation and next steps

- Retain the supplied CSVs locally; the label rule has been verified against their actual labels.
- Extend the completed same-split baseline comparison with training-fold-only tuning and an untouched evaluation partition.
- Select hyperparameters and thresholds without using the final evaluation partition.
- Treat feature importance as model association, not a causal explanation of sharing behavior.

![Random Forest feature importance](results/figures/random_forest_feature_importance.png)
![XGBoost feature importance](results/figures/xgboost_feature_importance.png)
