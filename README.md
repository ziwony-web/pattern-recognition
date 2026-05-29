# Online News Popularity Prediction

## 1. Project Overview
This project aims to predict whether an online news article will become popular based on its metadata and content-related features.
The task was formulated as a binary classification problem, where an article is classified as popular if it receives more than 1,400 shares.

The project compares two tree-based machine learning models, Random Forest and XGBoost, and analyzes their performance using Accuracy, F1 Score, and AUC.
The final model was selected based not only on validation performance but also on generalization ability on the test dataset.

## 2. Team Members
- Yunwoo Choi, Seojeong Cho, Ingyeong Yang, Jiwon Kang

## 3. Project Objective
- Dataset: 22,200 samples with 46 features (approx. 10% missing values).
- Target Variable: y=1 (Popular, >1,400 shares), y=0 (Not popular)
- Goal: Build a binary classification model to predict news popularity.

## 4. Key Features
### Topic and Category Features
- data_channel
- weekday
- is_weekend
- LDA_00, LDA_01, LDA_02, LDA_03, LDA_04
These features represent the article category, publication timing, and topic distribution.

### Visual and Structural Features
- num_imgs
- num_videos
- n_tokens_title
- n_tokens_content
- average_token_length
These features describe the article’s length, structure, and visual content.

### Keyword-related Features
- kw_avg_avg
- kw_avg_max
- kw_avg_min
- kw_max_avg
- kw_min_avg
These features reflect keyword quality, trend relevance, and the potential shareability of the article.

## 5. Data Preprocessing

The preprocessing pipeline included the following steps:

### Missing Value Handling
- Sentiment-related missing values were filled with 0.0, assuming neutral sentiment.
- Missing values in image and video count features were filled with 0.
- Missing values in numerical features were filled using the median.
- Missing values in weekday were filled with the mode.
- Missing values in data_channel were replaced with Other.
- 
### Categorical Encoding

Categorical variables such as weekday, data_channel, and is_weekend were encoded using one-hot encoding.
This allowed the model to use categorical information without imposing artificial ordinal relationships.

### Outlier Handling

For the XGBoost model, numerical columns were clipped using the 1st and 99th percentiles to reduce the effect of extreme outliers.

### Feature Removal

The following columns were removed because they were not useful for model training or were unavailable in the test set:

- id
- shares
- url
- timedelta

## 6. Models
Two tree-based machine learning models were implemented and compared.

### Random Forest Classifier

Random Forest is an ensemble model that combines multiple decision trees using a bagging strategy.
It is robust to overfitting and provides useful feature importance values.

The model was tuned using GridSearchCV with 3-fold cross-validation.

Main hyperparameters
- n_estimators
- max_depth
- min_samples_split
- max_features

The classification threshold was also adjusted from the default value of 0.5 to improve F1 Score.

### XGBoost Classifier

XGBoost is a gradient boosting model that builds trees sequentially, where each tree corrects the errors of the previous trees.
It provides strong predictive performance and includes regularization techniques to reduce overfitting.

The model was tuned using 5-fold cross-validation.

Main hyperparameters:
- min_child_weight
- learning_rate
- n_estimators
- reg_alpha
- reg_lambda
- max_depth
- gamma

The decision threshold was adjusted from 0.5 to 0.414 to improve performance under class imbalance.

## 7. Results
| Model | Accuracy | F1 Score | AUC | Final Average Score |
|---|---:|---:|---:|---:|
| Random Forest | 0.6788 | 0.6846 | 0.7520 | 0.7051 |
| XGBoost | 0.6587 | 0.7001 | 0.7243 | 0.6944 |

Although Random Forest achieved a higher validation score, its performance decreased noticeably on the test dataset.  
XGBoost showed a smaller gap between validation and test performance, indicating better generalization ability.

Therefore, XGBoost was selected as the final model.

## 8. Feature Importance Analysis

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

## 9. Key Findings
- Topic-related features and keyword scores were highly important for predicting news popularity.
- Sentiment and subjectivity features also contributed to model performance.
- Threshold tuning was useful for improving classification performance, especially F1 Score.
- XGBoost showed stronger generalization than Random Forest despite a slightly lower validation score.
- Evaluating only training or validation performance can be misleading when selecting the final model.
  
## 10. Limitations

This project had several limitations:

- Only the original features were used, with limited feature engineering.
- Class imbalance was not fully addressed.
- Interaction features such as kw_avg_avg × global_subjectivity were not included.
- Advanced interpretability methods such as SHAP or LIME were not applied.
- Deep learning-based models were not explored.

  
## 11. Future Improvements

Future work could include:

- Creating interaction features between keyword, topic, and sentiment variables
- Using clustering-based features to group similar article types
- Applying automated threshold optimization using ROC or precision-recall curves
- Using ensemble methods such as Voting or Stacking
- Applying SHAP or LIME for model interpretability
- Exploring deep neural networks for complex nonlinear feature interactions

## 12. Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Jupyter Notebook

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
