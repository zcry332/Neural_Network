# Banking Fraud Detection Project Overview
_This project focuses on building a Feedforward Neural Network (FFNN) model to effectively detect fraudulent transactions using a credit card transaction dataset. The core challenge lies in addressing the highly imbalanced nature of the data to create a model that is both sensitive to fraud (high recall) and accurate in its predictions (high precision)._

## Dataset Summary
### The dataset is extremely imbalanced, typical of real-world fraud detection problems:
- Total Transactions: 283,726 (after cleaning)
- Fraud (1) Samples: 473 (only about 0.17% of the data)
- Features: 30 anonymized numerical features, including _Time_ and _Amount_.

## Key Preprocessing Steps
1. Data Cleaning: 1081 duplicate rows were identified and removed.
2. Outlier Handle:
   Using a weighted class with a standard scaler to handle outliers, since most of the outliers in the dataset are non-fraud transactions, the neural network should be able to recognize these anomalies and distinguish the true fraud pattern.
3. Data Preparation:
   - Features were Standard Scaled using StandardScaler(), Robust Scaled using RobustScaler() in version 2.
   - The data was split using stratified sampling to ensure the train and test sets maintained the rare fraud cases.
  
## Model & Training Strategy
A dense Feedforward Neural Network (FFNN) with multiple hidden layers was constructed and trained using a strategy specifically designed to tackle the severe class imbalance:
### Recalibrated Version 1 model:
- Model Architecture:\
  A Sequential model with three Dense layers (64, 32, 16 hidden neurons) and 1 output layer, avoiding over-engineering.
- Imbalance Handling:\
  Class Weighting was used during training, with the fraud class (Class 1) assigned a weight of approximately 300.24. This forced the model to heavily penalize misclassifying fraud cases.
- Callbacks:\
  EarlyStopping and ReduceLROnPlateau were used to ensure efficient and stable training.

### Version 2 model :
- Model Architecture:\
  A Sequential model with three Dense layers (64, 32, 16 hidden neurons) and 1 output layer, with cross-validation applied during training.
  Applied batch normalization and L2 regularization at each hidden layer which using the LeakyRelU function.
  Additionally, use a 30% dropout to avoid over-fitting.
- Imbalance handle & Data Preparation::
  Using RobustScaler to be insensitive to outliers, since SMOTE is used to oversample the data, the model should learn from it on its own.
- Callbacks:\
  Still applied EarlyStopping and ReduceLROnPlateau, but since tbe 30% dropout rate, its effect would be limited.
  

## Model Performance on Test Set
### The recalibrated model's performance on the test set revealed a successful prioritization of fraud detection:
|      |precision|recall|f1-score|support|
|------|---------|------|--------|-------|
|1     |  0.44   | 0.80 |  0.57  |     95|
| 0    |1.00     | 1.00 |   1.00 |  56651|
|accuracy|       |      | 1.00   |  56746|
|macro avg|0.72   | 0.90|  0.79 |56746|
|weighted avg|  1.00 |1.00 |1.00   |  56746|

Confusion Matrix:
 |76| 19|
 |---|---|
 |95 |56556|

  _The loss on training and validation data:_
<img width="1018" height="525" alt="download" src="https://github.com/user-attachments/assets/0ca00dee-f938-485c-89e7-783ce98c7a67" />

 

### The re-engineered new version of the model has a better performance in recognising the FP with a relatively similar potency on catching the fraud:
Classification Report:
| |precision |recall |f1-score |support|
|------------|-----|-----|-----|-------|
|1  | 0.72  | 0.79  |0.75  |95|
|0  | 1.00  |1.00  |1.00|56651|
|accuracy  |||1.00  | 56746|
|macro avg  | 0.86  | 0.89 |0.88 |56746|
|weighted avg | 1.00 | 1.00 | 1.00  |56746|
Confusion Matrix:
 |75| 20|
 |---|---|
 |29 |56622|

_The loss on training and validation data:_
<img width="1010" height="528" alt="download (1)" src="https://github.com/user-attachments/assets/7d339e54-75e0-4161-8ab3-96209e3f5cf9" />

The high Recall (80%ish) confirms the model's effectiveness in identifying the majority of fraud cases. However, the resulting low Precision (44%) in model version 1 indicates a high rate of False Positives, which pulls the overall F1 score (0.57) down. Version 2 performed slightly better in terms of precision, yet it still fell short one perfection.
It reminds me that for continuous numeric features, random forest is the most robust and stable one. Therefore, I have an idea to ensemble version 2 with RandomForestClassifier to achieve a better result, and I achieved an AUPRC of 0.870178199798558. 
| |precision |recall |f1-score |support|
|------------|-----|-----|-----|-------|
| 1 |0.96  | 0.78  | 0.86  | 95 |
| 0  | 1.00 | 1.00 | 1.00  | 56651|
|accuracy ||| 1.00| 56746|
|macro avg | 0.98 | 0.89  |0.93 | 56746|
|weighted avg | 1.00 | 1.00 |1.00 |56746|

Confusion Matrix:
|74 |21|
|---|---|
|3| 56648|

So far, this is the best result within my capabilities to deliver; however, I still wonder if there is any other way to handle this type of data. It is a fun journey.


