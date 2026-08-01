import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, fbeta_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ==========================================
# 1. GENERATE SYNTHETIC CREDIT DATASET
# ==========================================
# Simulating a realistic 90:10 imbalanced dataset (90% Non-Default, 10% Default)
print("📦 Generating realistic credit dataset...")
np.random.seed(42)
n_samples = 10000

data = pd.DataFrame({
    'annual_inc': np.random.lognormal(mean=11, sigma=0.5, size=n_samples),
    'total_debt': np.random.exponential(scale=15000, size=n_samples),
    'credit_limit': np.random.uniform(5000, 50000, size=n_samples),
    'credit_balance': np.random.uniform(1000, 45000, size=n_samples),
    'delinq_2yrs': np.random.poisson(lam=0.3, size=n_samples),
    'emp_length_years': np.random.randint(0, 11, size=n_samples)
})

# Create a realistic target variable based on risk factors
risk_score = (data['total_debt'] / data['annual_inc'] * 2) + (data['credit_balance'] / data['credit_limit'] * 1.5) + (data['delinq_2yrs'] * 0.5)
# Set threshold to get roughly 10% defaults
data['is_default'] = (risk_score > np.percentile(risk_score, 90)).astype(int)

print(f"Dataset Shape: {data.shape}")
print(f"Class Distribution:\n{data['is_default'].value_counts(normalize=True)}\n")

# ==========================================
# 2. FEATURE ENGINEERING (Domain Ratios)
# ==========================================
print("🛠️ Engineering domain-specific financial ratios...")
data['debt_to_income'] = data['total_debt'] / data['annual_inc']
data['credit_utilization'] = data['credit_balance'] / data['credit_limit']

# Drop the highly correlated raw columns to prevent multicollinearity
X = data.drop(columns=['is_default', 'total_debt', 'credit_balance'])
y = data['is_default']

# Split Data into Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. FIX CLASS IMBALANCE USING SMOTE
# ==========================================
print("⚖️ Applying SMOTE to handle minority class (Defaults)...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"Original training shape: {y_train.value_counts().to_dict()}")
print(f"Resampled training shape: {y_train_resampled.value_counts().to_dict()}\n")

# ==========================================
# 4. TRAIN ADVANCED XGBOOST MODEL
# ==========================================
print("🚀 Training XGBoost Classifier...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train_resampled, y_train_resampled)

# ==========================================
# 5. BUSINESS & TECHNICAL EVALUATION
# ==========================================
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("📊 --- MODEL PERFORMANCE REPORT ---")
print(classification_report(y_test, y_pred))

# Calculate Financial Specific Metrics (F2 Score weights Recall higher than Precision)
f2 = fbeta_score(y_test, y_pred, beta=2)
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"Business-Critical F2-Score (Recall Intensive): {f2:.4f}\n")

# Generate and Display Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Default', 'Default'], yticklabels=['Non-Default', 'Default'])
plt.title('Credit Risk Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

# Save the plot inside your VS Code workspace directory
plt.savefig('confusion_matrix.png')
print("💾 Confusion matrix saved successfully as 'confusion_matrix.png'!")