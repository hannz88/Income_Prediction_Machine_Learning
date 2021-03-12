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
*RandomForestClassifier on the left; LogisticRegression on the right* 

The overall accuracy for both the models are the same but the precision and recall are different. Precision indicates the proportion of positive identifications that was actually correct; recall is showing the proportion of actual positives that were correctly identified. Precision uses false positive counts while recall uses false negative counts. As such, they're always at tug-of-war. Precision for `LogisticRegression` is higher but recall for `RandomForestClassifier` is higher. This situation isn't helped by the fact that we have an unbalanced dataset, the class 1 to class 0 is almost 1 to 16. As such, accuracy isn't sufficient enough to evaluate the model. f1 score isn't terribly high either. We'll look at other metrics to compare the models.

### ROC-AUC For Both Models
ROC stands for Receiver Operating Characistics that summarizes the trade-off between false positive rate and true positive rate (aka recall) using different probability thresholds for binary classifier. A classifier that only predicts the majority class under all thresholds will be represented by the line joining the bottom left to the upper right. The area under the curve (AUC) is helpful as they can be calculated and compared against other classifier. AUC of 1 means it's a perfect classifier.

<img src="https://github.com/hannz88/Income_Prediction_Machine_Learning/blob/main/Images/rocauc_RFC1.png" width="400"/> <img src="https://github.com/hannz88/Income_Prediction_Machine_Learning/blob/main/Images/rocauc_LR1.png" width="400"/>

Here, we can see that AUC for the `RandomForestClassifier` is 0.70 while AUC for `LogisticRegression` is 0.60 which is just a little lower but not a lot. The dummy model that does not learn has AUC of 0.5, as one would expect in a binary classifier.

### Precision-recall Curve
ROC-AUC isn't enough for imbalanced dataset. A few incorrect predictions could skew the results for a highly imbalanced dataset. An alternative is to use precision-recall curve, which focuses on the minority class. Much like ROC, the curve is calculated based on different thresholds and the AUC could be used to compare models. A dummy classifier will be a horizontal line on the plot with a precision that is proportional to the number of positive examples in the dataset.

<img src="https://github.com/hannz88/Income_Prediction_Machine_Learning/blob/main/Images/precision_recall_RFC1.png" width="400"/> <img src="https://github.com/hannz88/Income_Prediction_Machine_Learning/blob/main/Images/precision_recall_LR1.png" width="400"/>

In the figure above, `RandomForestClassifier` has AUC of 0.55 while `LogisticRegression` has AUC of 0.53 which, again, is slightly lower. 

## Grid search
I used `GridSearchCv` from `sklearn` for hyperparameter tuning. The following are the hyperparameters for the models.

`RandomForestClassifier`:
```
rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [200,500,700],
    'max_depth' : [2,4,6,8]
}
grid_search = GridSearchCV(rfc, param_grid=param_grid, cv=5, scoring = "f1")
grid_search.best_params_
>>> {'max_depth': 8, 'n_estimators': 700}
```

`LogisticRegression`:
```
lr2 = LogisticRegression(random_state=42, solver="liblinear")  # have to use liblinear as l1 is not supported by lbfgs
grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}
grid_lr = GridSearchCV(lr2, param_grid = grid_values,scoring = 'f1')
grid_lr.best_params_
>>> {'C': 10, 'penalty': 'l1'}
```

### Classifcation Report
<img src="https://github.com/hannz88/Income_Prediction_Machine_Learning/blob/main/Images/Classification_report_rfc2.png" width="400"/> <img src="https://github.com/hannz88/Income_Prediction_Machine_Learning/blob/main/Images/Classification_report_LR2.png" width="400"/>

From the classication report, we could see t

### ROC-AUC
<img src="https://github.com/hannz88/Income_Prediction_Machine_Learning/blob/main/Images/rocauc_GridSearch_RFC2.png" width="400"/> <img src="https://github.com/hannz88/Income_Prediction_Machine_Learning/blob/main/Images/rocauc_GridSearch_LR2.png" width="400"/>

### Precision-recall AUC
<img src="https://github.com/hannz88/Income_Prediction_Machine_Learning/blob/main/Images/precision_recall_RFC2.png" width="400"/> <img src="https://github.com/hannz88/Income_Prediction_Machine_Learning/blob/main/Images/precison_recall_GridSearch_LR2.png" width="400"/>
