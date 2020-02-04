# Mod_5_project_Banking_fraud

* Student name: Joshua Owusu Ankomah
* School name:  Flatiron School
* Project deadline 
* Instructor name: Dan Sanz

### Project Overview
In this project I trained several models to detect fraud transactions. I run 5 baseline models , LogisticRegression, KNeighborsClassifier, RandomForestClassifier, XGBClassifier, SupportVectorMachine Classifier. I continued to optimize top two models based on their train and test accuracy result. XGBoost and RandomForest Models. I did five iterations including grid search on hyperparameters, balancing the labels by SMOTE and subsampling from the original dataset. Both RandomForest and XGBoost model had over 99% accuracy on the data that includes all frauds and some random safe data. The data was still imbalanced so I did SMOTE over this dataset as well. At the end of those iterations, **XGBoost model had 99% accuracy** on both train and test sets.  
**Final Iterations saw the performance of both Random Forest and XGBoost rise to 100% accuracy**


### Project Steps

- 1.Loading Data and EDA
- 2.Feature Engineering
- 3.Machine Learning
    - 3.1. Baseline Models
    - 3.2. Grid Search for Best Hyper-parameter
    - 3.3. Dealing with Unbalanced Data
    - 3.3.1. Balancing Data via Resambling with SMOTE
    - 3.3.2. Subsampling Data from the Original Dataset
    - 3.3.3 Performing SMOTE on the New Data
    - 3.3.3a HyperParameter Optimization And Visualization of ROC_Curve of Models
             RandomForestModel and XGBoost Model

- 4.Machine Learning Pipeline
- 5.Feature Importance
- 6.Conclusion
- 7.Future Works


### Data
https://www.kaggle.com/ntnu-testimon/paysim1

**Variables in the columns of the Dataset:**

**step** - maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).

**type** - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.

**amount** - amount of the transaction in local currency.

**nameOrig** - customer who started the transaction

**oldbalanceOrg** - initial balance before the transaction

**newbalanceOrig** - new balance after the transaction

**nameDest** - customer who is the recipient of the transaction

**oldbalanceDest** - initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).

**newbalanceDest** - new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).

**isFraud** - This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.

**isFlaggedFraud** - The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction.



# Other Additions: Interpretation of classification reports

# Precision

- Precision is the ability of a classiifer not to label an instance positive that is actually negative. For each class it is defined as as the ratio of true positives to the sum of true and false positives. Said another way, “for all instances classified positive, what percent was correct?”

# Recall

- Recall is the ability of a classifier to find all positive instances. For each class it is defined as the ratio of true positives to the sum of true positives and false negatives. Said another way, “for all instances that were actually positive, what percent was classified correctly?”

# f1 score

- The F1 score is a weighted harmonic mean of precision and recall such that the best score is 1.0 and the worst is 0.0. Generally speaking, F1 scores are lower than accuracy measures as they embed precision and recall into their computation. As a rule of thumb, the weighted average of F1 should be used to compare classifier models, not global accuracy.

# support

- Support is the number of actual occurrences of the class in the specified dataset. Imbalanced support in the training data may indicate structural weaknesses in the reported scores of the classifier and could indicate the need for stratified sampling or rebalancing. Support doesn’t change between models but instead diagnoses the evaluation process.


#Conclusion:

### Accuracy results after iterations
I created a model that can predict fraud transactions. I used XGBoost and RandomForest  classifiers in this model. 

      (Data & Parameters)                              (Accuracy)   XGBoost   RandomForest
      **Iteration 1**                                                            
    - Random Sample & default parameters                                100%       83%                **Iteration 2** 
    - Random Sample & best parameters                                   85.5%     84.3%
      **Iteration 3**
    - Balanced data with SMOTE & best parameters                        99.4%     98.7% 
      **Iteration 4**
    - Random Safe trans. data and all Fraud data & best parameters      98.8%     99.6%
      **Iteration 5**
    - New data balanced with SMOTE & best parameters                    99%       92.1%

    - New Balanced data with SMOTE & best HpyerParametr Tunning .       100%      100%


























