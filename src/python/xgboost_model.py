import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, log_loss
from xgboost import XGBClassifier

# -----------------------------
# 0) Load data
# -----------------------------
CSV_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET = "Churn"

df = pd.read_csv(CSV_PATH)

# Drop ID column
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# Target Yes/No -> 1/0
df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)

# Convert TotalCharges to numeric
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

y = df[TARGET].values
X_df = df.drop(columns=[TARGET])

# Identify numeric vs categorical
num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_df.columns if c not in num_cols]

# -----------------------------
# 1) Split (train/val/test)
# -----------------------------
X_train_df, X_temp_df, y_train, y_temp = train_test_split(
    X_df, y, test_size=0.30, random_state=42, stratify=y
)

X_val_df, X_test_df, y_val, y_test = train_test_split(
    X_temp_df, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# -----------------------------
# 2) Preprocess
# -----------------------------
numeric_pipe = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median"))]
)

categorical_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ],
    remainder="drop"
)

X_train = preprocess.fit_transform(X_train_df)
X_val = preprocess.transform(X_val_df)
X_test = preprocess.transform(X_test_df)

# -----------------------------
# 3) XGBoost model
# -----------------------------
pos = np.sum(y_train == 1)
neg = np.sum(y_train == 0)
scale_pos_weight = neg / max(pos, 1)

model = XGBClassifier(
    n_estimators=4000,
    learning_rate=0.02,
    max_depth=5,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    gamma=0.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=50,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"Best iteration (trees): {model.best_iteration}")

# -----------------------------
# 4) Evaluate
# -----------------------------
val_probs = model.predict_proba(X_val)[:, 1]
test_probs = model.predict_proba(X_test)[:, 1]

print("\n=== VALIDATION RESULTS ===")
print(f"Val AUC: {roc_auc_score(y_val, val_probs):.4f}")
print(f"Val log-loss: {log_loss(y_val, val_probs):.4f}")

print("\n=== TEST RESULTS ===")
print(f"Test AUC: {roc_auc_score(y_test, test_probs):.4f}")
print(f"Test log-loss: {log_loss(y_test, test_probs):.4f}")

# -----------------------------
# 5) Feature importance
# -----------------------------
feature_names = num_cols + cat_cols
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n=== TOP 10 FEATURES ===")
print(importance_df.head(10).to_string(index=False))
