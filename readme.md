# Customer Churn Prediction

## Introduction

- This project aims to predict customer churn of a online retail store. Customer churn refers to the loss of clients or customers. By predicting churn, businesses can take proactive measures to retain customers.

- Here we performed EDA to understand which features are casuing the customers to churn and also after modelling we took the help
of shap scores to analyse feature importance and interaction effects

## Dataset 

We have this dataset consisting of 5630 data points with 20 feature variables-:
- E Comm CustomerID Unique customer ID
- E Comm Churn Churn Flag
- E Comm Tenure Tenure of customer in organization
- E Comm PreferredLoginDevice Preferred login device of customer
- E Comm CityTier City tier
- E Comm WarehouseToHome Distance in between warehouse to home of customer
- E Comm PreferredPaymentMode Preferred payment method of customer
- E Comm Gender Gender of customer
- E Comm HourSpendOnApp Number of hours spend on mobile application or website
- E Comm NumberOfDeviceRegistered Total number of deceives is registered on particular customer
- E Comm PreferedOrderCat Preferred order category of customer in last month
- E Comm SatisfactionScore Satisfactory score of customer on service
- E Comm MaritalStatus Marital status of customer
- E Comm NumberOfAddress Total number of added added on particular customer
- E Comm Complain Any complaint has been raised in last month
- E Comm OrderAmountHikeFromlastYear Percentage increases in order from last year
- E Comm CouponUsed Total number of coupon has been used in last month
- E Comm OrderCount Total number of orders has been places in last month
- E Comm DaySinceLastOrder Day Since last order by customer
- E Comm CashbackAmount Average cashback in last month

We have split the data set into 80:10:10 ratio for Train Val and Test Datasets after performing basic Data Cleaning 

## Libraries and Languages Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Exploratory Data Analysis

- Have perfomed eda on all the feature variables individually 
- Performed bi-variate analysis for churn and other features

### Intresting Insights

- The dataset is heavily imbalanced with only 17% customers churning 
- I applied both Smote and adjuster loss functions to deal with imbalance
- Tenure seems to be affecting churn rate the most as it had the highest shap value and mutual information score
- Features like coupons used and ordercount were not detrimental in predicting the churn rate
- Featues like warehouse distances and complains also played significant role
- Performed statistical analysis tests like normlaity tests (eg. shapiro wilk, anderson darling ) and also performed tests to check if two variables are correlated for both categorical (eg CHI-squared test)
and numerical (eg. ANOVA and Kruskal Wallis Tests)

## Feature Engineering

- First Task was to deal with null values
- We Performed Little MCAR test to check if the data was Missing Completely at Random
- After Analysing the missing values we came to a conclusion that the data was Missing Not at Random
- MNAR makes sense as misssing variables were tenure,hours spent on app, warehouse to home distance which are features which users are not very comfortable in revealing
- We used KNN imputer with k=3 to fill the missing values
- We used Isolation forests to detect outliers and saw that they were not too bad to be removed from the dataset
- Also performed PCA to plot them in 2 dimensions but these two principle components were only able to explain 20 percent of the variance


## Modelling
- Before Modelling we obviously used one-hot encoder to fit it on training set and then transform each of the three datasets
- We also applied standard scaler to scale the datasets
### Modelling 1
- Here we have used just weightes loss functions instead of oversampling
- **Logistic Regression**: We started with logistic regression as our baseline model. After tuning the hyperparamters such as regularization paramter and types of solvers we got a F1 score of 0.55 and a recall of 0.79 on validation set
- **Random Forest**: Improved the f1-score to 0.87 showing that the relationship was non-linear. The recall too improved to 0.81 on validation set
- **Support Vector Machine (SVM)**: This Gave us the best results both in terms F1-score and Recall. F1-score was 0.95 and recall was also 0.952.
- **XG boost**:  Gave decent result with F1- Score of 0.80 and a recall of 0.89

#### Test Set Results

/Users/sudhanvasavyasachi/Desktop/Projects/Customer_Churn/utils/Screenshot 2025-03-16 at 5.23.07 PM.png

### Modelling 2

- Here instead of using Weighted scores we used SMOTE technique and observed that it didnt create much of a difference


## Conclusion

- We decided to stick to Support vector Classifier model with a true positive rate of 94 percent
- This model has the potential to reduce the churn rate from 17 percent to a mere 9 percent considering the retention strategy worked for only 50 percent of them.
- Additional Details
By following the above steps, we were able to  build a robust model to predict customer churn. This can help businesses take timely actions to retain customers and improve their overall profitability.


