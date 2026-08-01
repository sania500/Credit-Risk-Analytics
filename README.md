# Credit Risk & Default Prediction Engine

An end-to-end financial analytics pipeline built to predict the probability of loan defaults, enabling financial institutions to optimize risk-based pricing and protect capital.

## 📊 Key Business Outcomes
* **Severe Class Imbalance Handled**: Utilized SMOTE to balance the 90:10 non-default to default ratio.
* **Exceptional Recall (98.5%)**: Successfully captured **197 out of 200 actual defaults** using an optimized XGBoost Classifier, minimizing high-cost False Negatives.
* **Risk Pricing Strategy**: Features translated raw probabilities into structured low, medium, and high-risk tiers.

## 🛠️ Tech Stack
* **Language**: Python
* **Libraries**: Scikit-Learn, Imbalanced-Learn, XGBoost, Pandas, Numpy, Matplotlib, Seaborn
