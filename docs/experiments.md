# Experiment evidence

Source snapshot: `474e7c7cd90a8069a046f5432e112ee91b458f1b` (before portfolio maintenance). Cell numbers below are one-based positions in the notebook JSON, not execution counters.

| Source | Evidence |
|---|---|
| `results/Random_Forest_Classfier.ipynb`, cell 16 | Accuracy 0.6788, F1 0.6846, AUC 0.7520, mean 0.7051 at threshold 0.48 |
| `final code/code.ipynb`, cell 10 | Split sizes (16650, 59) and (5550, 59), including target column |
| `final code/code.ipynb`, cell 18 | Accuracy 0.6587, F1 0.7001, AUC 0.7243, mean 0.6944 |

## Conditions

Random Forest: preprocessing followed by 80/20 split, seed 42; 3-fold GridSearchCV scored by F1; selected saved parameters: 200 trees, depth 22, min_samples_split 10, max_features log2. Thresholds 0.48, 0.50 and 0.52 were inspected on the same validation set.

Final XGBoost: preprocessing followed by 75/25 split, seed 100; shuffled 5-fold KFold; F1 grid search; depth 5 and gamma 0.3; threshold 0.414. Its displayed grid contains 2 × 2 × 3 × 3 × 3 = 108 parameter combinations. The final full-data submission cell specifies learning_rate 0.2, min_child_weight 7, 50 estimators, reg_alpha 0 and reg_lambda 100.

## Audit findings

- Imputation/clipping before splitting lets validation distribution influence preprocessing. Results need a leakage-controlled rerun.
- Final submission code recomputes imputation/clipping on test data and constructs dummy columns separately; reuse training-fitted transforms instead.
- The final submission cell scales features, while the preceding validation path does not use the same complete pipeline.
- Grid definitions and saved outputs in some exploratory XGBoost cells differ. Treat saved outputs as archival, not proof that the current cell source reproduces them.
- No populated external-test predictions or evaluation record supports the previous README's generalization-gap claim. That claim is omitted from the revised summary.
- Raw train/test CSVs are not committed. On 2026-10-14, supplied course CSVs were used for the new experiments below; historical scores above remain archival.

`results/historical_metrics.csv` transcribes saved values for convenient inspection. The new baseline uses a different preprocessing recipe and fixed threshold, so its new results are recorded separately.

## New experiments verified on 2026-10-14

Supplied train: 22,200 × 49; test: 9,515 × 47. Removing id/shares/y leaves 46 predictors. All labels agree with `shares > 1400`; 901 rows equal 1,400 and are labeled 0. Classes: 11,196 negative / 11,004 positive, so this is not a strongly imbalanced dataset.

Both runs use the same validation rows and data hash, stratified 80/20 split, seed 42 and threshold 0.5. Preprocessing is fitted only on the training partition. RF: 200 trees, depth 22, min_samples_split 10, max_features log2. XGBoost: 50 trees, depth 5, learning_rate 0.2, gamma 0.3, min_child_weight 7, reg_alpha 0, reg_lambda 100. These fixed settings came from archival work, so the comparison is not an untouched final test or a newly optimized competition result.

- random_forest: accuracy 0.6653153153, F1 0.6684515841, ROC AUC 0.7238999123, mean 0.6858889372.
- xgboost: accuracy 0.6671171171, F1 0.6663656885, ROC AUC 0.7243219869, mean 0.6859349308.

The small AUC difference does not support a model-superiority claim. Full-data refits also generated 9,515 unlabeled test predictions per model; these outputs are not test evaluation results. See [run evidence](../results/verified-2026-10-14/README.md) for hashes and versions.

The supplied [team Notion page](https://app.notion.com/p/Pattern-Recognition-1e5533ba738a80618330e3709b4e1bbe) could not be accessed through the connected workspace during this audit. Its contents have not been used as evidence.
