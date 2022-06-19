# Credit-Risk-Analysis

## Overview of the Analysis

The purpose of this analysis is to develop a machine learning model that effectively predicts high risk loans.
   
The model is trained with historical loan data including loan size, interest rate, borrower income, debt to income ratio, number of accounts, derogatory marks, total debt and current loan status. Our model uses current loan status as the label to classify high risk loans.  
  
During initial examination of the dataset, using 'value_counts', it was found the data consisted of 75036 loans with a status of 0 (non-risk loan) and 2500 loans with a status of 1 (high risk loan). Since the historical dataset had labels, a supervised learning approach was used for analysis.
    
Analysis began with data preparation. After retrieving the data a DataFrame was created for the data features (X) (loan size, interest rate, borrower income, debt to income ratio, number of accounts, derogatory marks, total debt). A seperate DataFrame was made for the labels (y).
    
The data was split into a training set and a testing set. The training dataset includes 75% of the historical data which is used to build and train the model. The remaining 25% of the data is the testing set of used to validate the model.
    
Effectiveness of the model can be assessed by using the trained model to make predictions on the test data. The predictions of the trained model can be compared to the actual values of the test data. The results of this comparison can be analyzed to determine the accuracy and overall effectiveness of the model. 
    
The analysis utilizes sklearn's LogisticRegression method for the first model. The model is trained with the original dataset. The first iteration of the model was found to have a large imbalance between the two classes. It was found that our data has far more non-risky loans (75036) than high risk loans (2500).
    
Due to the large imbalance, an additional iteration of the model is used. For the second model, imblearn's RandomOverSampler method was used. This method randomly over-sampled the class with less values (high-risk loans) to make it the same size as the other class. The result is a larger training dataset with an equal number of high-risk and non-risky loans. 
    
Model performance was evaluated by calculating the accuracy score, generating a confusion matrix and printing the classification report. This provided key metrics for model analysis such as precision, recall and F1 score.

## Results

Description of the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1: original dataset with class imabalance.
 - Accuracy Score: 0.952
 
  | Class | Precision | Recall |
  |-------|-----------|--------|
  |   0   |  1.00     |  0.99  |
  |   1   |  0.85     |  0.91  |
  


* Machine Learning Model 2: dataset with random oversampling.
 - Accuracy Score: 0.9937

  | Class | Precision | Recall |
  |-------|-----------|--------|
  |   0   |  1.00     |  1.00  |
  |   1   |  1.00     |  0.91  |

## Summary

Results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.


After using oversampling, we were able to increase model accuracy by 4.2%. 

For non-risky loans, Class 0, the precision of the model was 100% meaning all of the non-risky loan predictions were classified correctly. Recall for class 0 increased to 100% meaning the oversampled model correctly classified all of the actual non-risky loans as non-risky. 

For high risk loans, Class 1, the precision of the oversampled model was 100%. The oversampled model predicted 100% of high risk loans correctly. This is a 15% improvement of prevision compared to the original model. Recall for class 1 remained at 91%. 

The oversampled model is much better at correctly predicting high risk loans. The oversampled model was also very good at predicting non-risky loans as well. 

Since it is extremely important to accurately predict high risk loans. It is recommended to use the oversampled model to classify and predict loan status. 

---

## Technologies

This project leverages python 3.7 with the following packages:

* [pandas](https://github.com/pandas-dev/pandas)

* [anaconda](https://docs.anaconda.com/)

* [jupyter-lab](https://jupyterlab.readthedocs.io/en/stable/)

* [hvplot](https://pyviz-dev.github.io/hvplot/user_guide/Introduction.html)

* [scikit-learn](https://scikit-learn.org/stable/)

* [imbalance-learn](https://imbalanced-learn.org/stable/)

---

## Installation Guide
To install imbalanced-learn, execute the following command in your virtual environment prior to starting jupyter lab.
### imbalance-learn
```
conda install -c conda-forge imbalanced-learn
```

### Start Jupyter Lab
Once your conda virtural environment is started with all prerequisites, start Jupyter Lab:
```
jupyter lab
```

---

## Contributors


*  **Quintin Bland** <span>&nbsp;&nbsp;</span> |
<span>&nbsp;&nbsp;</span> *email:* quintinbland2@gmail.com <span>&nbsp;&nbsp;</span>|
<span>&nbsp;&nbsp;</span> [<img src="images/LI-In-Bug.png" alt="in" width="20"/>](https://www.linkedin.com/in/quintin-bland-a2b94310b/)

---

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)