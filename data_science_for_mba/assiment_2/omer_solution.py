# Final Submission - Omer Razi, ID: 304844798

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def build_preprocessor(df: pd.DataFrame):
    # Identify column types
    numeric_features = ['MonthlyCharges', 'TotalCharges', 'tenure', 'SeniorCitizen']
    categorical_features = [c for c in df.columns if df[c].dtype == 'object' and c not in ['customerID', 'Churn']]

    # Numeric pipeline: median impute + standardize
    num_pipeline = Pipeline([
        ('impute_num', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])

    # Categorical pipeline: mode impute + one-hot
    cat_pipeline = Pipeline([
        ('impute_cat', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine into a single transformer
    preprocessor = ColumnTransformer([
        ('nums', num_pipeline, numeric_features),
        ('cats', cat_pipeline, categorical_features)
    ], remainder='drop')

    return preprocessor


def evaluate_models(X_tr, y_tr, X_vl, y_vl):
    # Trains multiple models and returns the one with highest F1 on validation.
    candidates = {
        'LogisticReg': LogisticRegression(max_iter=500, random_state=304844798),
        'DecisionTree': DecisionTreeClassifier(random_state=304844798),
        'RandomForest': RandomForestClassifier(n_estimators=150, random_state=304844798),
        'KNN': KNeighborsClassifier()
    }
    scores = {}
    for name, clf in candidates.items():
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_vl)
        score = f1_score(y_vl, preds)
        print(f"Model: {name} (F1 = {score:.2f})")
        scores[name] = (score, clf)

    best_name = max(scores, key=lambda n: scores[n][0])
    print(f"--- Best model on validation: {best_name} (F1 = {scores[best_name][0]:.2f})")
    print()
    return scores[best_name][1]


if __name__ == "__main__":
    # Load raw data
    df_raw = pd.read_csv('churn_training.csv')
    df_hold = pd.read_csv('churn_holdout.csv')

    # Basic cleaning: coerce to numeric then fill with median
    for col in ['MonthlyCharges', 'TotalCharges', 'tenure', 'SeniorCitizen']:
        if col in df_raw:
            coerced = pd.to_numeric(df_raw[col], errors='coerce')
            med = coerced.median()
            df_raw[col] = coerced.fillna(med)
        if col in df_hold:
            coerced_h = pd.to_numeric(df_hold[col], errors='coerce')
            med_h = coerced_h.median()
            df_hold[col] = coerced_h.fillna(med_h)

    # Fill missing categoricals in training
    cat_fill = [
        'Partner', 'Dependents', 'InternetService',
        'DeviceProtection', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
    for c in cat_fill:
        if c in df_raw:
            df_raw[c].fillna(df_raw[c].mode()[0], inplace=True)

    # Map target to numeric
    df_raw['ChurnLabel'] = df_raw['Churn'].map({'No': 0, 'Yes': 1})
    df_raw.drop(columns=['Churn'], inplace=True)

    # Split train vs validation
    features = df_raw.drop(columns=['customerID', 'ChurnLabel'])
    target = df_raw['ChurnLabel']
    X_train, X_val, y_train, y_val = train_test_split(
        features, target,
        test_size=0.2,
        random_state=304844798,
        stratify=target
    )

    # Build and fit preprocessor on combined data (to capture all categories)
    prepr = build_preprocessor(pd.concat([features, df_hold], axis=0))
    X_train_p = prepr.fit_transform(X_train)
    X_val_p = prepr.transform(X_val)

    # Model evaluation
    best_clf = evaluate_models(X_train_p, y_train, X_val_p, y_val)

    # Retrain best model on full training set
    X_full_p = prepr.transform(features)
    best_clf.fit(X_full_p, target.values)

    # Prepare holdout and predict
    hold_features = df_hold.drop(columns=['customerID', 'Churn'], errors='ignore')
    X_hold_p = prepr.transform(hold_features)
    hold_preds = best_clf.predict(X_hold_p)

    # Save submission CSV
    submission = pd.DataFrame({
        'CustomerID': df_hold['customerID'],
        'Churn_Prediction': hold_preds.astype(int)
    })
    submission.to_csv('304844798.csv', index=False)
    print("Created '304844798.csv' with holdout predictions.")