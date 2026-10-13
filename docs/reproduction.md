# Reproduction and data recovery

## Required inputs

Restore the actual course `train.csv` and `test.csv` from the original course materials or your own backup. Existing CSV files in this repository are two-byte placeholders, including `final code/prediction.csv`; they are not datasets or model predictions.

The training CSV must have binary `y` labels plus predictors. The new baseline excludes `y`, `shares`, `id`, `url`, and `timedelta` from predictors. It accepts numeric and categorical columns, rejects missing/nonbinary labels, and does not reconstruct labels from shares. Preserve the assignment's exact target convention.

The public UCI dataset is not a drop-in replacement: the course data has custom labels, missing values, and schema/splits. Do not claim exact reproduction after silently substituting it.

## Environment

`requirements.txt` contains the core packages for the new baseline. Its preprocessing and script execution were smoke-checked with Python's installed scikit-learn 1.8.0 and pandas 2.2.3. Historical dependency versions are not fully known. XGBoost and Jupyter are optional for inspecting/running the original notebooks.

## Outputs and verification

Run the command in the root README. Each new run writes `metrics.json` (data hash, versions, split, feature list, threshold, scores) and `validation_predictions.csv`. Do not commit private input data by default. Independently rerun before adding new scores to the README.

The original notebooks retain their paths and outputs for reference. Their Google Drive paths need local adjustment; install `xgboost` and Jupyter separately to explore them. Do not treat their execution counters or saved outputs as a clean top-to-bottom execution record.
