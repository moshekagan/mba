# Churn Prediction - Final Pipeline with Logistic Regression

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Step 1:
# Load data
training_df = pd.read_csv("churn_training.csv")
holdout_df = pd.read_csv("churn_holdout.csv")

for col in ['MonthlyCharges', 'TotalCharges', 'tenure', 'SeniorCitizen']:
    # Convert column to numeric, coerce errors
    training_df[col] = pd.to_numeric(training_df[col], errors='coerce')
    holdout_df[col] = pd.to_numeric(holdout_df[col], errors='coerce')
    # Fill missing numeric values with median
    training_df[col].fillna(training_df[col].median(), inplace=True)
    holdout_df[col].fillna(holdout_df[col].median(), inplace=True)

# Fill missing categorical values with most frequent
categorical_cols_fill = ['Partner', 'Dependents', 'InternetService', 'DeviceProtection', 'StreamingMovies', 'Contract', 'PaymentMethod']
for col in categorical_cols_fill:
    training_df[col].fillna(training_df[col].mode()[0], inplace=True)

# Feature/Target separation and split for evaluation
X = training_df.drop(columns=["customerID", "Churn"])
y = training_df["Churn"].map({"Yes": 1, "No": 0})
X_holdout = holdout_df.drop(columns=["customerID", "Churn"])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("cat", categorical_pipeline, categorical_cols),
    ("num", numeric_pipeline, numeric_cols)
])

# Step 2: Train model on full data
X_full = training_df.drop(columns=["customerID", "Churn"])
y_full = training_df["Churn"].map({"Yes": 1, "No": 0})

X_full_processed = preprocessor.fit_transform(X_full)
X_val_processed = preprocessor.transform(X_val)
X_holdout_processed = preprocessor.transform(X_holdout)

final_model = LogisticRegression(max_iter=1000)
final_model.fit(X_full_processed, y_full)

# Step 5: Make predictions and export CSV
holdout_preds = final_model.predict(X_holdout_processed)
submission = pd.DataFrame({
    "CustomerID": holdout_df["customerID"],
    "Prediction_Churn": holdout_preds
})
submission.to_csv("submission.csv", index=False)

# Step 6: Evaluate F1 on validation set
val_preds = final_model.predict(X_val_processed)
val_f1 = f1_score(y_val, val_preds)

print(f"\n📊 The model predicted {submission['Prediction_Churn'].sum()} out of {len(submission)} customers will churn.")
print(f"✅ F1 Score on validation set: {val_f1:.4f}")

# Explanation:
# F1 = 2 * (precision * recall) / (precision + recall)
# It balances false positives and false negatives and is effective for imbalanced classification problems.
