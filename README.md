# Online News Popularity Prediction


## Team Members
- Yunwoo Choi, Seojeong Cho, Ingyeong Yang, Jiwon Kang

## Project Overview
- This project has developed a binary classification model that predicts whether or not an online news article will be popular based on the metadata and content characteristics of it.

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

## 5. Models
- Logistic Regression
- Random Forest
- XGBoost
- SVM

## 6. Results
| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression |  |  |  |  |  |
| Random Forest |  |  |  |  |  |
| XGBoost |  |  |  |  |  |

## 7. Key Findings
- 어떤 feature가 결과에 영향을 주었는지
- 어떤 모델이 가장 좋았는지
- 왜 그런 결과가 나왔는지

## My Contribution
- This project was conducted as a team project for a Pattern Recognition course.
My contributions included participating in data preprocessing, model experimentation, performance comparison, and report preparation.

Main contributions:

- Participated in preprocessing strategy design
- Compared Random Forest and XGBoost model performance
- Analyzed feature importance results
- Contributed to the final report and discussion section

## 9. Limitations & Future Work
This project had several limitations

- Only the original features were used, with limited feature engineering.
- Class imbalance was not fully addressed.
- Interaction features such as kw_avg_avg × global_subjectivity were not included.
- Advanced interpretability methods such as SHAP or LIME were not applied.
- Deep learning-based models were not explored.

## 10. Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- XGBoost
