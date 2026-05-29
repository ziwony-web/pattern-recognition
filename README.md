# Online News Popularity Prediction

## Project Overview
This project aims to predict whether an online news article will become popular based on its metadata and content-related features.
The task was formulated as a binary classification problem, where an article is classified as popular if it receives more than 1,400 shares.

The project compares two tree-based machine learning models, Random Forest and XGBoost, and analyzes their performance using Accuracy, F1 Score, and AUC.
The final model was selected based not only on validation performance but also on generalization ability on the test dataset.

## Team Members
- Yunwoo Choi, Seojeong Cho, Ingyeong Yang, Jiwon Kang

## Project Objective
- Dataset: 22,200 samples with 46 features (approx. 10% missing values).
- Target Variable: y=1 (Popular, >1,400 shares), y=0 (Not popular)
- Goal: Build a binary classification model to predict news popularity.


## Methodology
1. 데이터 수집
2. 데이터 전처리
3. EDA
4. 모델 학습
5. 성능 평가
6. 결과 해석

## Models
- Logistic Regression
- Random Forest
- XGBoost
- SVM

## Results
| Model | Accuracy | F1 Score | AUC | Final Average Score |
|---|---:|---:|---:|---:|
| Random Forest | 0.6788 | 0.6846 | 0.7520 | 0.7051 |
| XGBoost | 0.6587 | 0.7001 | 0.7243 | 0.6944 |

Although Random Forest achieved a higher validation score, its performance decreased noticeably on the test dataset.  
XGBoost showed a smaller gap between validation and test performance, indicating better generalization ability.

Therefore, XGBoost was selected as the final model.

## Feature Importance Analysis

Feature importance analysis showed that the following types of features played an important role in predicting article popularity:

Keyword score features such as kw_avg_avg, kw_avg_max, and kw_min_avg
Topic distribution features such as LDA_02, LDA_01, and LDA_00
Sentiment-related features such as global_subjectivity, global_sentiment_polarity, and global_rate_positive_words
Lexical diversity features such as n_unique_tokens

These results suggest that article popularity is influenced not only by article length or format, but also by semantic topic patterns, emotional tone, and keyword quality.

### Random Forest Feature Importance

![Random Forest Feature Importance](results/figures/random_forest_feature_importance.png)

### XGBoost Feature Importance

![XGBoost Feature Importance](results/figures/xgboost_feature_importance.png)

## Key Findings
- 어떤 feature가 결과에 영향을 주었는지
- 어떤 모델이 가장 좋았는지
- 왜 그런 결과가 나왔는지

## My Contribution
This project was conducted as a team project for a Pattern Recognition course.
My contributions included participating in data preprocessing, model experimentation, performance comparison, and report preparation.

Main contributions:

- Participated in preprocessing strategy design
- Compared Random Forest and XGBoost model performance
- Analyzed feature importance results
- Contributed to the final report and discussion section

## Limitations & Future Work
This project had several limitations

- Only the original features were used, with limited feature engineering.
- Class imbalance was not fully addressed.
- Interaction features such as kw_avg_avg × global_subjectivity were not included.
- Advanced interpretability methods such as SHAP or LIME were not applied.
- Deep learning-based models were not explored.

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- XGBoost
