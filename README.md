# Model-based Credit Risk Evaluator

This repo represents an example of utilizing supervised machine learning to make predictions. Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. Data is resampled and fed into a Logistic Regression machine learning model.  Comparisons are then made on the accuracy, precision, and recall between the original imbalanced data and resampled data.

## Overview of the Analysis

### Objective

The objective of this predictive modeling was to create a model that could correctly classify high-risk loans based on a certain set of data features.

### Approach

The approach taken included the following six steps:

1. **Read the data**: A CSV file was provided with the following features: loan size, loan interest rate, borrower income,debt-to-income ratio, number of accounts open, derogatory credit report marks, and total debt.
2. **Split the data into training and testing sets**: The data was split into training and testing sets with 75% being allocated to training and the remaining 25% allocated to testing.  Splits were accomplished using scikit-learn's train_test_split object.
3. **Oversample the data**:  The data was significantly imbalanced with 75,036 records representing healthy loans and only 2,500 representing high-risk loans.  An oversampling technique was chosen using the RandomOverSampler object in the imblearn library.
4. **Fit a logistic regression model**: Despite being called logisitic *regression*, the model is used for classification problems. No other models were tested in this analysis since the evaluation of the predictive outcomes were positive.
5. **Make predictions**: Predictions were made using the the test data created in step #2.
6. **Evaluate the model**: The model was evaluated using functionality in the sci-kit learn (sklearn) and imblanced learn (imblearn) libraries.  The results included calculated metrics for overall model accuracy, precision, and recall.  

## Results

The balanced accuracy reports, confusion reports, and the consolidated summary of each offered in the classification report yielded the following results:

* Machine Learning Model 1: *Logistic Regression with original imbalanced data*

  * **Accuracy**: The accuracy measures how often the model was correct.  In this case the model has a 95% accuracy rating, which is very high and builds confidence.
  * **Precision**: The precision, also known as the positive predictive value (PPV), measures how confident we are that the model correctly made the positive predictions.  It measures the reliability of positive classification. The model was 100% reliable in measuring healthy loans and 85% reliable in identifying high-risk loans.  These numbers are high and also build confidence.
  * **Recall**: Recall measures the amount of actual healthy loans that the model classified as healthy as well as the actual number of high-risk loans that the model classified as high-risk. With 99% and 91% rates, respectively, the model is performing well.

* Machine Learning Model 2: *Logistic Regression with randomly oversampled data*

  * **Accuracy**: The model has a 99% accuracy rating, which is very high and builds confidence.
  * **Precision**: The model was 99% reliable in measuring healthy loans and 84% reliable in identifying high-risk loans.  These numbers are high and also build confidence.
  * **Recall**: The model correctly classified 99% rates of both healthy and unhealthy loans.

## Summary

Overall, both logistic regression model do a great job of predicting both healthy and high-risk loans. The resampling of the imbalanced increased the amount of high-risk loans properly classified by 8% at the expense of reducing the amount of properly classified healthy loans by 1%.  Identification of high-risk loans are what the business objective of the model is, so the model using resampled data would the best one to use.
