
# News Popularity Prediction
> Binary classification of online news articles using Random Forest & XGBoost  
> Pattern Recognition Course · Team Project

---

## Overview

Built a binary classification model to predict whether a news article will be shared more than 1,400 times, using the UCI Online News Popularity dataset. Compared Random Forest and XGBoost through systematic experiments and selected the final model based on **generalization performance** rather than validation score alone.

- **Dataset**: 22,200 samples / 46 features (~10% missing values)
- **Target**: `y=1` (shares > 1,400), `y=0` (otherwise)
- **Evaluation**: Average of Accuracy, F1 Score, and AUC

---

## Results

| Model | Accuracy | F1 Score | AUC | Avg Score |
|---|:-:|:-:|:-:|:-:|
| Random Forest | 0.6788 | 0.6846 | 0.7520 | 0.7051 |
| **XGBoost (Final)** | **0.6587** | **0.7001** | **0.7243** | **0.6944** |

Random Forest achieved a higher validation score, but showed a significant drop on the test set (0.705 → 0.662). XGBoost had a smaller gap between training and test performance, demonstrating more stable generalization — and was selected as the final model.

> **Key Insight**: Selecting a model based solely on training performance risks overfitting. Generalization ability must be evaluated alongside validation metrics.

---

## Preprocessing

| Step | Details |
|---|---|
| Missing values | Sentiment features → `0.0` (neutral); image/video counts → `0`; numerical → median; `weekday` → mode; `data_channel` → `'Other'` |
| Categorical encoding | `weekday`, `data_channel`, `is_weekend` → One-Hot Encoding |
| Outlier handling | Numerical columns clipped to 1–99th percentile *(XGBoost only)* |
| Dropped columns | `id`, `url`, `timedelta`, `shares` |

---

## Modeling

### Random Forest
- Hyperparameter search via **GridSearchCV** with 3-fold cross-validation
- Best params: `n_estimators=200`, `max_depth=22`, `max_features='log2'`, `min_samples_split=10`
- Classification threshold adjusted from 0.5 → **0.48** to improve F1 Score

### XGBoost
- Hyperparameter search via **5-fold cross-validation**
- Key params: `max_depth=5`, `gamma=0.3`, L1/L2 regularization applied
- Classification threshold adjusted from 0.5 → **0.414** to handle class imbalance

---

## Feature Importance

Both models consistently ranked the following feature groups as most important:

| Feature Group | Examples |
|---|---|
| Keyword quality | `kw_avg_avg`, `kw_avg_max`, `kw_min_avg` |
| Topic distribution | `LDA_00`, `LDA_01`, `LDA_02` |
| Sentiment indicators | `global_subjectivity`, `global_sentiment_polarity` |
| Lexical diversity | `n_unique_tokens` |

→ **Topic relevance, emotional tone, and keyword quality** are stronger predictors of shareability than article length or structure.

### Random Forest Feature Importance

![Random Forest Feature Importance](results/figures/random_forest_feature_importance.png)

### XGBoost Feature Importance

![XGBoost Feature Importance](results/figures/xgboost_feature_importance.png)
---

## My Contribution

- Participated in designing the preprocessing strategy (missing value handling)
- Ran experiments comparing Random Forest and XGBoost; analyzed performance trade-offs
- Interpreted feature importance results across both models
- Contributed to the final report and Discussion section

---

## Limitations & Future Work

- Limited feature engineering (e.g., no interaction terms like `kw_avg_avg × global_subjectivity`)
- Class imbalance not fully addressed
- Model interpretability tools (SHAP, LIME) not applied
- Ensemble methods (Voting, Stacking) not explored


## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=flat&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

