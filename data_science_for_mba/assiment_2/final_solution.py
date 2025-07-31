# Final assigment - Churn Prediction
# Moshe Kagan
# 200830842

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load and clean the data
training_df = pd.read_csv("churn_training.csv")
holdout_df = pd.read_csv("churn_holdout.csv")

for col in ['MonthlyCharges', 'TotalCharges', 'tenure', 'SeniorCitizen']:
    # Convert to numeric, coerce errors
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

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=67, stratify=y)

# Preprocessing pipeline
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

# Transform data for model comparison
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

# Compare models using F1 score
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier()
}

print("\nModel Comparison on Validation Set:")
results = {}
for name, model in models.items():
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_val_processed)
    f1 = f1_score(y_val, y_pred)
    results[name] = f1
    print(f"{name}: F1 Score = {f1:.4f}")

# Final model training on full dataset
X_full = training_df.drop(columns=["customerID", "Churn"])
y_full = training_df["Churn"].map({"Yes": 1, "No": 0})

X_full_processed = preprocessor.fit_transform(X_full)
X_holdout_processed = preprocessor.transform(X_holdout)

# Get the best model based on F1 score and the model name
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name} with F1 Score = {results[best_model_name]:.3f}")

final_model = best_model
final_model.fit(X_full_processed, y_full)

# Predictions and export CSV
holdout_preds = final_model.predict(X_holdout_processed)
submission = pd.DataFrame({
    "CustomerID": holdout_df["customerID"],
    "Churn_Prediction": holdout_preds
})
submission.to_csv("200830842.csv", index=False)

# Evaluate F1 on validation set
val_preds = final_model.predict(X_val_processed)
val_f1 = f1_score(y_val, val_preds)

print(f"\nThe model predicted {submission['Churn_Prediction'].sum()} out of {len(submission)} customers will churn.")
print(f"Final Logistic Regression F1 Score on validation set: {val_f1:.4f}")