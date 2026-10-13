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
- The repository does not contain real train/test CSVs. No new course-data score has been produced.

`results/historical_metrics.csv` transcribes saved values for convenient inspection. The new baseline uses a different preprocessing recipe and fixed threshold, so its future results must be recorded separately.
