# Fraud Detection Model Comparison

## Overview
This project evaluates the performance of various machine learning models for detecting fraudulent transactions. The dataset used contains credit card transactions with features like time, amounts, and anonymized transaction details. Models such as Random Forest, Decision Tree, Logistic Regression, K-Nearest Neighbors (KNN), and Support Vector Machines (SVM) with both RBF and Linear kernels were trained and evaluated using precision, recall, F1 score, ROC-AUC score, and accuracy metrics.

## Dataset
The dataset consists of the following columns:
- *Time*: Time elapsed between transactions
- *V1-V28*: Anonymized features representing transaction details
- *Amount*: Transaction amount
- *Class*: Binary label (1 for fraudulent, 0 for non-fraudulent)

## Model Performance Comparison

### Performance Metrics

| Model                        | Precision (Class 1) | Recall (Class 1) | F1 Score (Class 1) | ROC-AUC Score | Accuracy |
|------------------------------|---------------------|-------------------|---------------------|---------------|----------|
| **Random Forest**            | 0.93                | 0.89              | 0.91                | 0.947         | 1.00     |
| **Decision Tree**            | 0.49                | 0.81              | 0.61                | 0.904         | 1.00     |
| **Logistic Regression**      | 0.06                | 0.95              | 0.11                | 0.961         | 0.97     |
| **K-Nearest Neighbors (KNN)**| 0.51                | 0.88              | 0.64                | 0.94          | 1.00     |
| **SVM with RBF Kernel**      | 0.10                | 0.91              | 0.18                | 0.94          | 0.98     |
| **SVM with Linear Kernel**   | 0.06                | 0.91              | 0.12                | 0.94          | 0.97     |

### Performance Metric Definitions

- **Precision (Class 1)**: Precision is the ratio of correctly predicted positive observations to the total predicted positives. High precision indicates low false positive rate.
- **Recall (Class 1)**: Recall is the ratio of correctly predicted positive observations to the all observations in actual class. High recall indicates low false negative rate.
- **F1 Score (Class 1)**: The F1 score is the harmonic mean of precision and recall, providing a single metric that balances both concerns.
- **ROC-AUC Score**: Receiver Operating Characteristic - Area Under the Curve (ROC-AUC) summarizes the trade-off between the true positive rate and false positive rate for a predictive model using different probability thresholds.
- **Accuracy**: Overall accuracy of the model, which may be affected by class imbalance.

### Insights and Recommendations

1. **Random Forest**:
   - **Precision**: Highest (0.93), indicating few false positives.
   - **Recall**: Very high (0.89), identifying most fraudulent transactions.
   - **F1 Score**: Best balance (0.91) between precision and recall.
   - **ROC-AUC Score**: Excellent (0.947), strong discriminative ability.
   - **Accuracy**: Influenced by class imbalance, but high (1.00).

2. **Decision Tree**:
   - **Precision**: Moderate (0.49).
   - **Recall**: High (0.81), good at detecting fraudulent transactions.
   - **F1 Score**: Moderate (0.61).
   - **ROC-AUC Score**: Good (0.904), decent discriminative ability.
   - **Accuracy**: Also influenced by imbalance, but high (1.00).

3. **Logistic Regression**:
   - **Precision**: Very low (0.06), high false positives.
   - **Recall**: Very high (0.95), detects almost all fraudulent transactions.
   - **F1 Score**: Very low (0.11), due to poor precision.
   - **ROC-AUC Score**: Highest (0.961), excellent discriminative ability.
   - **Accuracy**: Lower (0.97) due to precision issues.

4. **K-Nearest Neighbors (KNN)**:
   - **Precision**: Moderate (0.51).
   - **Recall**: High (0.88), effective in identifying fraud.
   - **F1 Score**: Moderate (0.64).
   - **ROC-AUC Score**: Excellent (0.94), good discriminative ability.
   - **Accuracy**: Influenced by imbalance, high (1.00).

5. **SVM with RBF Kernel**:
   - **Precision**: Low (0.10), high false positives.
   - **Recall**: Very high (0.91), identifies most fraudulent cases.
   - **F1 Score**: Low (0.18), due to poor precision.
   - **ROC-AUC Score**: Good (0.94), solid discriminative ability.
   - **Accuracy**: High (0.98), despite precision issues.

6. **SVM with Linear Kernel**:
   - **Precision**: Very low (0.06), many false positives.
   - **Recall**: Very high (0.91), detects most fraudulent transactions.
   - **F1 Score**: Very low (0.12), due to poor precision.
   - **ROC-AUC Score**: Good (0.94), effective discriminative ability.
   - **Accuracy**: Lower (0.97), influenced by precision challenges.

### SMOTE (Synthetic Minority Over-sampling Technique)

To address class imbalance in the dataset, Synthetic Minority Over-sampling Technique (SMOTE) was employed. SMOTE generates synthetic samples for the minority class (fraudulent transactions) to balance the class distribution, thereby improving model performance in detecting fraud.

### Confusion Matrix Analysis

#### Random Forest Confusion Matrix
|            | Predicted Fraudulent | Predicted Non-Fraudulent |
|------------|----------------------|--------------------------|
| **Actual Fraudulent**     | 91                   | 12                       |
| **Actual Non-Fraudulent** | 7                    | 56852                    |

- **True Positives (TP)**: 91
- **False Positives (FP)**: 7
- **True Negatives (TN)**: 56852
- **False Negatives (FN)**: 12

#### Logistic Regression Confusion Matrix
|            | Predicted Fraudulent | Predicted Non-Fraudulent |
|------------|----------------------|--------------------------|
| **Actual Fraudulent**     | 91                   | 12                       |
| **Actual Non-Fraudulent** | 1617                 | 55242                    |

- **True Positives (TP)**: 91
- **False Positives (FP)**: 1617
- **True Negatives (TN)**: 55242
- **False Negatives (FN)**: 12

#### SVM with RBF Kernel Confusion Matrix
|            | Predicted Fraudulent | Predicted Non-Fraudulent |
|------------|----------------------|--------------------------|
| **Actual Fraudulent**     | 91                   | 12                       |
| **Actual Non-Fraudulent** | 851                  | 56008                    |

- **True Positives (TP)**: 91
- **False Positives (FP)**: 851
- **True Negatives (TN)**: 56008
- **False Negatives (FN)**: 12

Based on the model performance and confusion matrices:
- **Random Forest** demonstrates the best overall performance with high precision, recall, and F1 score, effectively minimizing false positives and negatives.
- **Logistic Regression** and **SVM models** (RBF and Linear kernels) show high recall but suffer from very low precision, resulting in significant false positives.
- **Decision Tree** and **KNN** offer moderate performance, with KNN slightly outperforming Decision Tree in precision and F1 score.
- **Recommendation**: For balanced performance and effective fraud detection, **Random Forest** is recommended due to its robust metrics across all evaluation criteria.

#Random Forest Classification Graphical Visualization
<img width="644" alt="image" src="https://github.com/Asim-Vinayak-ML-Projects/Credit-Card-Fault-Detection/assets/140016882/05cfb8dd-c5ae-4623-925f-53b76f1ac2af">

This scatter plot represents the predicted probabilities of transactions being fraudulent. Each point corresponds to a transaction, and the y-axis shows the predicted probability of fraud. The red dashed line represents the decision threshold. Transactions above this threshold are classified as fraudulent.

<img width="641" alt="image" src="https://github.com/Asim-Vinayak-ML-Projects/Credit-Card-Fault-Detection/assets/140016882/11efb178-d5ac-4263-8cb7-aee9899d376d">

The feature importance plot shows the significance of each feature used by the Random Forest model in making predictions. Features are ranked by their importance scores, with `V14`, `V12`, `V4`, and `V10` being the most significant.

## Conclusion
The evaluation of machine learning models for fraud detection in credit card transactions highlights **Random Forest** as the optimal choice. Its ability to maintain high precision while detecting a significant portion of fraudulent transactions makes it suitable for real-world applications where minimizing false positives is crucial. Further improvements could focus on addressing class imbalance and refining feature engineering to enhance model performance.

## Installation and Setup
### Prerequisites
- Python 3.x
- Jupyter Notebook
- Necessary libraries: pandas, numpy, matplotlib, scikit-learn

### Setup
1. Clone the repository:
   ```bash
   git clone "https://github.com/your_username/FraudDetection.git"
   cd FraudDetection

2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib scikit-learn

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
