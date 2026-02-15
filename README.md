# Machine Learning Assignment 2 - Spam-Detection-App

## a. Problem Statement
The objective of this App is to classify emails as **Spam** (1) or **Not Spam / Ham** (0). By analyzing the frequency of specific words and characters in emails, the model aims to automatically filter unwanted messages. This is a classic binary classification problem with significant real-world application in cybersecurity and user experience.

## b. Dataset Description
* **Dataset Name:** Spambase Data Set
* **Source:** UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/94/spambase)
* **Type:** Binary Classification
* **Instances:** 4601.
* **Features:** 57.
* **Target Variable:**
    * `1`: Spam
    * `0`: Not Spam
* **Feature Details:** The dataset contains 57 continuous attributes. Most attributes indicate the frequency of a particular word or character in the email.

## c. Models Used & Comparison Table
The following six classification models were implemented and evaluated.

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9197 | 0.9713 | 0.9317 | 0.8744 | 0.9021 | 0.8353 |
| **Decision Tree** | 0.9099 | 0.9079 | 0.9137 | 0.8692 | 0.8909 | 0.8150 |
| **KNN** | 0.8936 | 0.9452 | 0.8989 | 0.8436 | 0.8704 | 0.7814 |
| **Naive Bayes** | 0.8208 | 0.9419 | 0.7193 | 0.9462 | 0.8173 | 0.6714 |
| **Random Forest** | 0.9533 | 0.9842 | 0.9753 | 0.9128 | 0.9430 | 0.9050 |
| **XGBoost** | 0.9566 | 0.9883 | 0.9730 | 0.9231 | 0.9474 | 0.9114 |

## d. Observations
Observations on the performance of each model on the Spambase dataset:

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Strong baseline performance with ~92% accuracy. The high AUC (0.9713) suggests it is excellent at ranking probabilities, even if it assumes a linear relationship between features. |
| **Decision Tree** | Good accuracy (~91%) but slightly lower AUC (0.9079) compared to ensemble models. It provides high interpretability but is prone to overfitting and variance. |
| **KNN** | The lowest accuracy among the top tier (89.36%). While decent, it struggled slightly compared to tree-based models, likely due to the high dimensionality (57 features) affecting distance calculations. |
| **Naive Bayes** | **Highest Recall (0.9462)** but lowest Precision (0.7193). It is excellent at catching spam but flags too many legitimate emails as spam (False Positives). |
| **Random Forest (Ensemble)** | Excellent all-around performance (95.33% Accuracy). It effectively reduced the variance of single decision trees and achieved a very high precision (0.9753). |
| **XGBoost (Ensemble)** | **The Best Performer.** Achieved the highest Accuracy (95.66%) and AUC (0.9883). Its gradient boosting technique effectively learned complex patterns in the word frequency data. |

---

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SachinRathee-bits/Spam-Detection-App.git
    cd Spam-Detection-App
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Training Script:**
    This will train the models and save them to the `model/` folder.
    ```bash
    python model/train_model.py
    ```
4.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```