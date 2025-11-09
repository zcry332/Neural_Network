# Banking Fraud Detection Project Overview
_This project focuses on building a Feedforward Neural Network (FFNN) model to effectively detect fraudulent transactions using a credit card transaction dataset. The core challenge lies in addressing the highly imbalanced nature of the data to create a model that is both sensitive to fraud (high recall) and accurate in its predictions (high precision)._

## Dataset Summary
### The dataset is extremely imbalanced, typical of real-world fraud detection problems:
- Total Transactions: 283,726 (after cleaning)
- Fraud (1) Samples: 473 (only about 0.17% of the data)
- Features: 30 anonymized numerical features, including _Time_ and _Amount_.

## Key Preprocessing Steps
1. Data Cleaning: 1081 duplicate rows were identified and removed.
2. Outlier Removal:
   An Isolation Forest model was applied to identify and remove 568 high-dimensional outliers using the most correlated features, aiming to improve the model's focus on the true fraud pattern.
3. Data Preparation:
   - Features were Standard Scaled using StandardScaler.
   - The data was split using stratified sampling to ensure the train and test sets maintained the rare fraud cases.
  
## Model & Training Strategy
A dense Feedforward Neural Network (FFNN) with multiple hidden layers was constructed and trained using a strategy specifically designed to tackle the severe class imbalance:
- Model Architecture:\
  A Sequential model with five Dense layers (258, 128, 64, 32, 1 neuron).
- Imbalance Handling:\
  Class Weighting was used during training, with the fraud class (Class 1) assigned a weight of approximately 300.24. This forced the model to heavily penalize misclassifying fraud cases.
- Callbacks:\
  EarlyStopping and ReduceLROnPlateau were used to ensure efficient and stable training.

## Model Performance on Test Set
The model's performance on the test set revealed a successful prioritization of fraud detection:\
|Metric|Score|
|--------------|--------|
|Test Accuracy|0.9889|
|Test AUC	|0.9150|
|Test AUPRC	|0.4924|

Classification Results (Threshold: 0.5):
|Class	|Precision	|Recall	|F1-Score|
|------------|-----|-----|-----|
|0 (Non-Fraud)	|1.00	|0.99	|0.99|
|1 (Fraud)	|0.11	|0.81	|0.20|

The high Recall (81%) confirms the model's effectiveness in identifying the majority of fraud cases (77 out of 95). However, the resulting low Precision (11%) indicates a high rate of False Positives (614 non-fraud cases incorrectly flagged as fraud), which pulls the overall F1-Score (0.20) down.
##Optimization Strategy: The Next Steps
To improve the balance between Recall and Precision (boosting the F1-Score) and make the model more practical for real-world use, the following optimization steps are recommended:
1. Adjust the Decision Threshold\
The current threshold of 0.5 leads to many False Positives. By raising the decision threshold, we can demand higher confidence from the model before it flags a transaction as fraudulent, thereby increasing Precision and reducing false alarms.
  - Action: Test thresholds (e.g., 0.6, 0.7, 0.8, 0.9) to find the optimal point that balances a high Recall (still important) with acceptable Precision.
2. Implement Data Resampling Techniques\
While class weighting was effective, Synthetic Minority Over-sampling Technique (SMOTE) or targeted undersampling can physically alter the training data distribution to provide the model with a clearer signal.
  - Action: Re-train the FFNN using a training set balanced via SMOTE (for example, targeting a 1:1 or 1:5 ratio of fraud to non-fraud cases) to see if it learns better boundary separation.
3. Explore Alternative Loss Functions\
The standard binary cross-entropy loss doesn't explicitly focus on hard-to-classify samples.
  - Action: Implement Binary Focal Loss, which dynamically scales the cross-entropy loss so that misclassified examples contribute more to the total loss, focusing the training on the most challenging fraud cases.
4. Enhance Model Stability and Generalization\
The model's architecture can be tuned with regularization layers.
  - Action: Re-introduce BatchNormalization() layers between Dense layers, and add Dropout layers to prevent overfitting, which may be beneficial given the small number of fraud examples.
