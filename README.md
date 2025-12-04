# Banking Fraud Detection Project Overview
_his project focuses on building a Feedforward Neural Network (FFNN) to detect fraudulent transactions in a highly imbalanced credit card dataset. The main challenge is to design a model that is both sensitive to fraud (high recall) and accurate in its alerts (high precision), while avoiding excessive false positives._

## Dataset Summary
### The dataset is extremely imbalanced, which is typical for real-world fraud detection:
- Total Transactions: 283,726 (after cleaning)
- Fraudulent samples (Class 1): 473 (~0.17% of all transactions)
- Legitimate samples (Class 0): 283,253
- Features: 30 anonymized numerical features, including _Time_ and _Amount_.

## Key Preprocessing Steps
1. Data Cleaning: Removed 1,081 duplicate rows, leaving 283,726 unique transactions.
2. Outlier Handle:
- Since most extreme values (outliers) are actually non-fraud transactions, the model needs to learn to ignore noisy high values while still detecting true fraud patterns.
- Experiments were run with:
   - StandardScaler (Version 1)
   - RobustScaler (Version 2), which is less sensitive to outliers that should be ignored
4. Data Preparation:
   - Used stratified splitting to keep the rare fraud ratio consistent between train and test sets.
	- For the more advanced version, used time-aware cross-validation (TimeSeriesSplit) to respect the chronological nature of transactions and avoid look-ahead leakage.
	- On the training data, experimented with:
   	* Class weighting (penalizing fraud misclassification more heavily)
   	* SMOTE oversampling for the minority class (fraud).
  
## Model & Training Strategy
### Version 1 – Recalibrated Baseline FFNN
- Model architecture
- Keras Sequential model with:
   - Dense(64) → Dense(32) → Dense(16) hidden layers
   - 1 sigmoid output neuron for binary classification
Kept architecture relatively simple to avoid over-engineering.
- Imbalance handling
   - Used class weighting during training:
   - The fraud class (Class 1) was given a much larger weight (~300.24), so misclassifying fraud had a much higher loss than misclassifying non-fraud.
- Scaling & splitting
   - Features scaled using StandardScaler.
   - Train/validation split used a simple stratified split.
- Callbacks
   - EarlyStopping to stop training when validation performance stopped improving.
   - ReduceLROnPlateau to lower the learning rate when the validation metric plateaued.

### Version 2 – Regularized FFNN with SMOTE & Time-Series CV
- Model architecture
   Still a dense FFNN with three hidden layers (64, 32, 16) and 1 output neuron.
   - Improvements:
      - Batch Normalization after each hidden layer
      - L2 regularization on weights
      - LeakyReLU activation for better gradient flow
      - Dropout (30%) to reduce overfitting
   - Imbalance handling & data preparation
      - Switched to RobustScaler to reduce the impact of outliers.
      - Used SMOTE on the training folds only to oversample the minority class (fraud).
      - Optionally combined with class weights to further emphasize fraud cases.
   - Cross-validation strategy
      - Used TimeSeriesSplit to create time-ordered train/validation folds:
      - Train on earlier transactions, validate on later ones.
      - This better simulates real-world deployment where we predict on future data.
   - Callbacks
      - Still used EarlyStopping and ReduceLROnPlateau.
      - Due to the heavier regularization (dropout + L2 + batch norm), the model stayed relatively stable with low validation loss across folds.


Model Performance on Test Set

Below, “Class 1” is fraud, and “Class 0” is legitimate.

Version 1 – Baseline FFNN (StandardScaler + Class Weights + BCE)

Classification report:

## Model Performance on Test Set
### The recalibrated model's performance on the test set revealed a successful prioritization of fraud detection:
|      |precision|recall|f1-score|support|
|------|---------|------|--------|-------|
|1     |  0.62   | 0.81 |  0.70 |     95|
| 0    |1.00     | 1.00 |   1.00 |  56651|
|accuracy|       |      | 1.00   |  56746|
|macro avg|0.81 | 0.90|  0.85 |56746|
|weighted avg|  1.00 |1.00 |1.00   |  56746|

Confusion Matrix:
 |77| 18|
 |---|---|
 |47 |566041|

AUPRC: 0.77152108479389


  _The loss on training and validation data:_
<img width="1010" height="525" alt="image" src="https://github.com/user-attachments/assets/848df52a-20b5-4356-ace9-726be17b8b02" />


 

### The re-engineered new version of the model has a better performance in recognising the FP with a relatively similar potency on catching the fraud:
Classification Report:
| |precision |recall |f1-score |support|
|------------|-----|-----|-----|-------|
|1  | 0.49  | 0.86  |0.63  |57|
|0  | 1.00  |1.00  |1.00|56689|
|accuracy  |||1.00  | 56746|
|macro avg  | 0.75  | 0.93 |0.81 |56746|
|weighted avg | 1.00 | 1.00 | 1.00  |56746|

Confusion Matrix:
 |49| 8|
 |---|---|
 |59 |56622|

AUPRC: 0.7355152784448674

_The loss on training and validation data:_
<img width="1001" height="526" alt="image" src="https://github.com/user-attachments/assets/65968787-c63b-4a1c-9ec9-d89145052847" />

## Final Ensemble: FFNN (Version 2) + RandomForestClassifier
To further improve robustness, especially for continuous numeric features (where tree-based models like RandomForest often shine), Version 2’s FFNN was ensembled with a RandomForestClassifier:
   - The FFNN v2 produced a probability score p_nn.
   - The RandomForest produced a probability score p_rf.
   - The final ensemble score was a weighted average:\
     $p_{\text{ensemble}} = 0.711 \cdot p_{\text{nn}} + 0.289 \cdot p_{\text{rf}}$
   
This ensemble is the best configuration found so far: it keeps high recall while remaining precision and AUPRC.

| |precision |recall |f1-score |support|
|------------|-----|-----|-----|-------|
| 1 |0.75  | 0.77  | 0.76  | 57 |
| 0  | 1.00 | 1.00 | 1.00  | 56689|
|accuracy ||| 1.00| 56746|
|macro avg | 0.87 | 0.89  |0.88 | 56746|
|weighted avg | 1.00 | 1.00 |1.00 |56746|

Confusion Matrix:
|44 |13|
|---|---|
|15| 56648|

AUPRC: 0.723118

So far, this is the best result I’ve been able to achieve:
   - Starting from a class-weighted FFNN,
   - Improving with RobustScaler + SMOTE + regularization + time-series CV,
   - And finally ensembling the FFNN with a RandomForestClassifier,
   - Reaching an AUPRC of ~0.87 with excellent precision and strong recall.

There are still other directions that could be explored for this kind of data, such as:
   - Trying gradient boosting models (XGBoost, LightGBM, CatBoost) and ensembling them with the NN.
   - Cost-sensitive tuning based on real business costs of false positives vs false negatives.
   - More advanced temporal modeling (e.g., sequence models over cardholder transaction histories).

Overall, it has been a fun and insightful journey into handling extremely imbalanced, time-dependent fraud data, and combining neural networks with tree-based models to get practical, high-quality fraud detection.


