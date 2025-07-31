# Churn Prediction - Final Pipeline with Logistic Regression
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Loading files
training_df = pd.read_csv("churn_training.csv")
holdout_df = pd.read_csv("churn_holdout.csv")

# Convert 'TotalCharges' to numeric, coerce errors
training_df['TotalCharges'] = pd.to_numeric(training_df['TotalCharges'], errors='coerce')
holdout_df['TotalCharges'] = pd.to_numeric(holdout_df['TotalCharges'], errors='coerce')

# Filling missing values in numeric columns with the median
for col in ['MonthlyCharges', 'TotalCharges', 'tenure', 'SeniorCitizen']:
    training_df[col].fillna(training_df[col].median(), inplace=True)
    holdout_df[col].fillna(holdout_df[col].median(), inplace=True)

# Filling missing values in categorical columns with the mode
categorical_cols = ['Partner', 'Dependents', 'InternetService', 'DeviceProtection',
                    'StreamingMovies', 'Contract', 'PaymentMethod']
for col in categorical_cols:
    training_df[col].fillna(training_df[col].mode()[0], inplace=True)

# Check for missing values
print(training_df.isnull().sum())
print(holdout_df.isnull().sum())

# Dependent variable is 'Churn', independent variables are all others except 'customerID'
X = training_df.drop(columns=["customerID", "Churn"])
y = training_df["Churn"].map({"Yes": 1, "No": 0})  # המרה לערכים בינאריים

X_holdout = holdout_df.drop(columns=["customerID", "Churn"])

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Select categorical and numeric columns
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

# Pipeline for categorical features
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Pipeline for numeric features
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# into a single preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("cat", categorical_pipeline, categorical_cols),
    ("num", numeric_pipeline, numeric_cols)
])

# Process the training, validation, and holdout sets
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_holdout_processed = preprocessor.transform(X_holdout)

# Step 2: Load and clean the data
training_df = pd.read_csv("churn_training.csv")
holdout_df = pd.read_csv("churn_holdout.csv")

# Convert 'TotalCharges' to numeric, coerce errors
training_df['TotalCharges'] = pd.to_numeric(training_df['TotalCharges'], errors='coerce')
holdout_df['TotalCharges'] = pd.to_numeric(holdout_df['TotalCharges'], errors='coerce')

# Fill missing numeric values with median
for col in ['MonthlyCharges', 'TotalCharges', 'tenure', 'SeniorCitizen']:
    training_df[col].fillna(training_df[col].median(), inplace=True)
    holdout_df[col].fillna(holdout_df[col].median(), inplace=True)

# Fill missing categorical values with most frequent
categorical_cols_fill = ['Partner', 'Dependents', 'InternetService', 'DeviceProtection',
                         'StreamingMovies', 'Contract', 'PaymentMethod']
for col in categorical_cols_fill:
    training_df[col].fillna(training_df[col].mode()[0], inplace=True)

# Step 3: Feature/Target separation and split for evaluation
X = training_df.drop(columns=["customerID", "Churn"])
y = training_df["Churn"].map({"Yes": 1, "No": 0})
X_holdout = holdout_df.drop(columns=["customerID", "Churn"])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Preprocessing pipeline
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

# Step 5: Train model on full data and generate predictions
X_full = training_df.drop(columns=["customerID", "Churn"])
y_full = training_df["Churn"].map({"Yes": 1, "No": 0})

X_full_processed = preprocessor.fit_transform(X_full)
X_val_processed = preprocessor.transform(X_val)
X_holdout_processed = preprocessor.transform(X_holdout)

final_model = LogisticRegression(max_iter=1000)
final_model.fit(X_full_processed, y_full)

# Step 6: Make predictions and export CSV
holdout_preds = final_model.predict(X_holdout_processed)
submission = pd.DataFrame({
    "CustomerID": holdout_df["customerID"],
    "Prediction_Churn": holdout_preds
})
submission.to_csv("submission.csv", index=False)

# Step 7: Evaluation on validation set
val_preds = final_model.predict(X_val_processed)
val_f1 = f1_score(y_val, val_preds)

print(f"📊 The model predicted {submission['Prediction_Churn'].sum()} out of {len(submission)} customers will churn.")
print(f"✅ F1 Score on validation set: {val_f1:.4f}")

# Explanation of F1:
# F1 = 2 * (precision * recall) / (precision + recall)
# It balances false positives and false negatives — good for imbalanced classes.
