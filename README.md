# A Machine Learning Project Predicting Income
[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)

## Introduction
This project was original presented by Reply Data during the Data Science Campus Challenge. The data used was US Census Data. However, year was not specified. The main objective was to predict if income of a person is more than $50k a year. The main techniques/ skills involved in the project are:

- Data exploration
- Data cleaning/ wrangling
- Feature Engineering
- Machine learning:
    - Model training/ fitting
    - Model evaluation using ROC/AUC
    - Grid search
    - Feature selection

## Data exploration
Original data dimension was 199523 rows x 23 columns. The variables are as follows:

<p align="center">
    <img src="https://github.com/hannz88/Income_Prediction_Machine_Learning/blob/main/Images/Variables.png" alt="Names of the variables">
</p>

While NA wasn't used to indicate missing missing values, the dataset used "?" or "Not in universe" fill in the null.

## Data cleaning & Feature Engineering
In order to simplify the data, a few of the categories were turned into binary data such as birth place of father, birth place of mother, etc. Wherever possible, 1 would indicate "United States". `Sex` was also binarized. Furthermore, the missing values were imputed using the mode for the respective columns. In order to normalize the distribution of the ratio data, `MinMaxScaler` was used while `OneHotEncoder` was applied to categorical data. These were used in conjunction with `Pipeline` and `ColumnTransformer`.

## Machine Learning
The processed data was then fitted with `RandomForestClassifier` and `LogisticRegression`. `RandomForestClassifier` is an ensemble ML model has generally been well-performing and reduces the risk of overfitting than using a single `DecisionTree`. As the outcome was binary, `LogisticRegression` is apt for the situation. The outcome from both models were compared.

### Classification Report For Both Models

<img src="https://github.com/hannz88/Income_Prediction_Machine_Learning/blob/main/Images/RFC1_class_report.png" width="400"/> <img src="https://github.com/hannz88/Income_Prediction_Machine_Learning/blob/main/Images/LR1_class_report.png" width="400"/> 

The overall accuracy for both the models are the same but the precision and recall are different. Precision indicates the proportion of positive identifications that was actually correct; recall is showing the proportion of actual positives that were correctly identified. Precision uses false positive counts while recall uses false negative counts. As such, they're always at tug-of-war. Precision for `LogisticRegression` is higher but recall for `RandomForestClassifier` is higher. This situation isn't helped by the fact that we have an unbalanced dataset, the class 1 to class 0 is almost 1 to 16.

### ROC-AUC For Both Models

<img src="https://github.com/hannz88/Income_Prediction_Machine_Learning/blob/main/Images/rocauc_RFC1.png" width="400"/> <img src="https://github.com/hannz88/Income_Prediction_Machine_Learning/blob/main/Images/rocauc_LR1.png" width="400"/>

