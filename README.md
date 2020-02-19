# Mod_5_project_Banking_fraud

* Student name: Joshua Owusu Ankomah
* School name:  Flatiron School
* Project deadline  
* Instructor name: Dan Sanz


# Context
There is a lack of public available datasets on financial services and specially in the emerging mobile money transactions domain. Financial datasets are important to many researchers and in particular to us performing research in the domain of fraud detection. Part of the problem is the intrinsically private nature of financial transactions, that leads to no publicly available datasets.

We present a synthetic dataset generated using the simulator called PaySim as an approach to such a problem. PaySim uses aggregated data from the private dataset to generate a synthetic dataset that resembles the normal operation of transactions and injects malicious behaviour to later evaluate the performance of fraud detection methods.


# Content
PaySim simulates mobile money transactions based on a sample of real transactions extracted from one month of financial logs from a mobile money service implemented in an African country. The original logs were provided by a multinational company, who is the provider of the mobile financial service which is currently running in more than 14 countries all around the world.

This synthetic dataset is scaled down 1/4 of the original dataset and it is created just for Kaggle.



# Project Steps
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



# Other Additions: Reasons for classification reports

# Precision
- Precision is the ability of a classiifer not to label an instance positive that is actually negative. For each class it is defined as as the ratio of true positives to the sum of true and false positives. Said another way, “for all instances classified positive, what percent was correct?”

# Recall
- Recall is the ability of a classifier to find all positive instances. For each class it is defined as the ratio of true positives to the sum of true positives and false negatives. Said another way, “for all instances that were actually positive, what percent was classified correctly?”

# f1 score
- The F1 score is a weighted harmonic mean of precision and recall such that the best score is 1.0 and the worst is 0.0. Generally speaking, F1 scores are lower than accuracy measures as they embed precision and recall into their computation. As a rule of thumb, the weighted average of F1 should be used to compare classifier models, not global accuracy.

# support
- Support is the number of actual occurrences of the class in the specified dataset. Imbalanced support in the training data may indicate structural weaknesses in the reported scores of the classifier and could indicate the need for stratified sampling or rebalancing. Support doesn’t change between models but instead diagnoses the evaluation process.




# Data
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




# Goal: 
- My goal is to classify the data according to fraud detection and create the best model that best detects fraud with the least
number of errors. I will be running through a couple of different models to see which one performs best for my data.



# EDA Findings and Preprocessing:


# 1.Observations¶

Distrubition plot shows number of transactions occured each hour (step). There are drastic changes in the number of transactions occuriing from time to time.


# 2.Observations :
 we can see that after 72 hours and 90 hours and after 400 hours during the month , the fraud transctions occur at a steady pace.

# 3.Observation :
* The plot indicates some sort of seasonality in the number of transaction during the day. A pattern is observed every 24 hour ;however , we do not know what time of the day fully represents "0" represents on the graph but a higher transaction clusters around the middle of the 24hour period.

# 4.Observations:
* We can see that the fraudlent transactions do not show a significant pattern as compared to the graph of the safe transactions in terms of number of occurence.
* The fraudulent tranactions occur almost every hour at the same frequency . There are more fraud transactions in low amounts and less in high amounts. However , there are no changes in the pattern.

# 5.Observations:
* We can see that the safe transactions are in very low amounts ;however , there is a peek in $1M but above that the frequency decreases

# 6.Observation:
There is an interesting peak on $1m. let's investigate a bit more on how many fraudent transactions happenning at $1m

# 7.Observation:
We can see that fraudulent cases of $1M occurred 287 times and this is the maximum amount.
Also there is an interesting amount of 0.00 which occured 16 times
Also a few amounts have been flagged as fraudulent transactions

# 8.Overall Observations:
Fraud transactions happen in large range from $119 to &10M. The frequency distribution of the amount of money involved in fraud transactions is positively skewed. Most of the fraud transactions are small amounts.
Majority of fraudulent transactions are lower than $1M . However in $1M there is an interesting increase similar to safe transactionsand that is also the maximum amount in all fraud transactions. There are also the maximum amount in all fraud transactions. There are also some fraud labelled transactions that have 0 amount. There are also 16 of these transactions.

# 9.Observations
We can see that all the fraudulent transactions are only with cash_outs and Trasnfers. Debit usage is very safe. Therefore it is best to use only the Cash_Out and Transfer Data for the model since the other type of transactions have no fraudulent activities involved.

# 10.Observations:
Only 16 transactions were postively flagged fraud and all of them happend to be Transfers. All positive values on the isFlaggedFraud are also positive on isFraud column. There are also inconsistencies in Origin and destination balances on these instances. Maybe that is why they were marked Fraud. That is valuable information to keep too.



# Summary of Observations
- I first discovered was that the number of transactions occured each hour and but there seemed to be some drastic changes in the number of transactions occuring from time to time.

- It was also observed that after 72 hours , 90 hours and after 400 hours during the month the fraud transactions occur at a steady pace.
Thus , eventhough safe transactions slows down in 3rd and 4th day and after 16th day of the month, fraud transactions happens at a steady pace. Especially in the second half of the month there are much less safe transactions but number of fraud transactions does not decrease at all. 

- Fraud proportion over all transactions is 0.01% while the fraud amount proportion is 0.1%

- There is some sort of seasonality in the number of transaction every 24 hours.Fraud transactions does not show that significant pattern. They happen every hour almost in the same frequency.

- There are more fraud transactions in low amounts and less in high amount. This distribution does not change much.

- Fraud transaction happens in a large range such as $119 dolars to  $10M dolars.
 Most of the fraud transactions are of Lesser amount. But in $1M there is an interesting increase similar to safe transactions. 

- There are 16 fake fraud cases  with '0' amount.

- Fraud activities only happens with TRANSFER and CASH_OUT transactions. DEBIT usage is very safe.




# Project Modelling Overview
- In this project I trained several models to detect fraud transactions after selecting my baseline model. 
  
- I run 5 main models , LogisticRegression, KNeighborsClassifier, RandomForestClassifier, XGBClassifier, SupportVectorMachine   Classifier. 

- I continued to optimize top two models based on their train and test accuracy result. XGBoost and RandomForest Models. 

- I did five iterations including grid search on hyperparameters, balancing the labels by SMOTE and subsampling from the     original dataset. 
  Both RandomForest and XGBoost model had over 99% accuracy on the data that includes all frauds and some random safe data. 
  
- The data was still imbalanced so I did SMOTE over this dataset as well. At the end of those iterations, **XGBoost model had 99% accuracy** on both train and test sets.  

- **Final Iterations saw the performance of both Random Forest and XGBoost rise to 100% accuracy**



# Feature Importance
### Observations : 
- We can see that the most important features of the random forest model is the OldBalanceDest ,NewBalanceDest and  Step
- XGBoost's most important features are newbalance , oldbalanceOrg and Step.
- Each Model gives different importance to the features ; however , oldbalanceOrg and newbalanceDest are the major indicators for both models




# Conclusion:
### Accuracy results after iterations
- I created a model that can predict fraud transactions. I used XGBoost and RandomForest  classifiers in this model. 

      (Data & Parameters)                               (Accuracy) XGBoost   RandomForest
 **Iteration 1**                                                            
    - Random Sample & default parameters                              100%      83%                
 
 **Iteration 2** 
    - Random Sample & best parameters                                 85.5%     84.3%       ****                   ***
 
 **Iteration 3**
    - Balanced data with SMOTE & best parameters                      99.4%     98.7% 
 
 **Iteration 4**
    - Random Safe trans. data and all Fraud data & best parameters    98.8%     99.6%
 
 **Iteration 5**
    - New data balanced with SMOTE & best parameters                  99%       92.1%
 
 **Final Iteration**
    - New Balanced data with SMOTE & best HpyerParametr Tuning .      100%      100%


























