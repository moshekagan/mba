import numpy as np
from numpy import dtype
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, make_scorer
from sklearn.pipeline import Pipeline

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 99999)
pd.set_option('max_colwidth', None)  # was -1


def calc_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    conf_mat = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("Confusion Matrix:")
    print(pd.DataFrame(conf_mat, columns=['Predicted No', 'Predicted Yes'], index=['Actual No', 'Actual Yes']))
    tn, fp, fn, tp = conf_mat.ravel()
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    return f1


def example(X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    # Make predictions
    y_pred = clf.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    f1 = calc_f1(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print(f'F1: {f1:.2f}')


def preprocess(df: pd.DataFrame, is_submission: bool = False) -> pd.DataFrame:
    for col in df.columns:
        if df[col].isnull().any():
            n_rows = df[col].isnull().sum()
            print(f'Filling NaN values in column: {col} -- {n_rows} rows')
            majority = df[col].mode()[0]
            df[col] = df[col].fillna(majority)

    gender_col = df['gender'].map({'Male': 1, 'Female': 0})
    senior_citizen_col = df['SeniorCitizen']  # already numeric, just convert to int
    partner_col = df['Partner'].map({'Yes': 1, 'No': 0})
    dependents_col = df['Dependents'].map({'Yes': 1, 'No': 0})
    tenure = df['tenure']  # numeric column, no need to convert
    phone_service_col = df['PhoneService'].map({'Yes': 1, 'No': 0})
    multiple_line_col = pd.get_dummies(df['MultipleLines'],
                                       prefix='MultipleLines') * 1  # 3 types: No, Yes, No phone service
    internet_service_col = pd.get_dummies(df['InternetService'],
                                          prefix='InternetService') * 1  # 3 types: DSL, Fiber optic, No
    online_security_col = pd.get_dummies(df['OnlineSecurity'],
                                         prefix='OnlineSecurity') * 1  # 3 types: No, Yes, No internet service
    online_backup_col = pd.get_dummies(df['OnlineBackup'],
                                       prefix='OnlineBackup') * 1  # 3 types: No, Yes, No internet service
    device_protection_col = pd.get_dummies(df['DeviceProtection'],
                                           prefix='DeviceProtection') * 1  # 3 types: No, Yes, No internet service
    tech_support_col = pd.get_dummies(df['TechSupport'],
                                      prefix='TechSupport') * 1  # 3 types: No, Yes, No internet service
    streaming_tv_col = pd.get_dummies(df['StreamingTV'],
                                      prefix='StreamingTV') * 1  # 3 types: No, Yes, No internet service
    streaming_movies_col = pd.get_dummies(df['StreamingMovies'],
                                          prefix='StreamingMovies') * 1  # 3 types: No, Yes, No internet service
    contract_col = pd.get_dummies(df['Contract'], prefix='Contract') * 1  # 3 types: Month-to-month, One year, Two year
    paperless_billing_method_col = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
    payment_method_col = pd.get_dummies(df['PaymentMethod'], prefix='PaymentMethod') * 1
    monthly_charges = df['MonthlyCharges']  # numeric column, no need to convert
    total_charges = df['TotalCharges'].str.replace(' ', '0').astype(float)  # numeric column, no need to convert
    mean_value = total_charges[total_charges != 0].mean()
    total_charges = total_charges.replace(0, mean_value)  # Replace 0 with mean value for TotalCharges

    if not is_submission:
        churn = df['Churn'].map({'Yes': 1, 'No': 0})  # Convert target variable to binary
    else:
        churn = None

    new_cols = [
        gender_col, senior_citizen_col, partner_col, dependents_col, tenure, phone_service_col, multiple_line_col,
        internet_service_col,
        online_security_col, online_backup_col, device_protection_col, tech_support_col, streaming_tv_col,
        streaming_movies_col,
        contract_col, paperless_billing_method_col, payment_method_col, monthly_charges, total_charges,
    ]
    if not is_submission:
        new_cols.append(churn)

    df_pp = pd.concat(new_cols, axis=1)
    return df_pp


def get_knn_pipeline() -> Pipeline:
    pipe = Pipeline(
        [
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ])
    return pipe


def create_knn_grid_search(X_train, y_train, cv: int = 5) -> GridSearchCV:
    pipe = get_knn_pipeline()  # Create pipeline

    # Define parameter grid (note the 'knn__' prefix)
    param_grid = {
        'knn__n_neighbors': [3, 5, 7],
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2]  # Manhattan (p=1) and Euclidean (p=2)
    }
    f1 = make_scorer(f1_score)
    grid = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring=f1, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid


def get_lr_pipeline() -> Pipeline:
    pipe = Pipeline(
        [
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(max_iter=1000))
        ])
    return pipe


def create_logistic_regression_grid_search(X_train, y_train, cv: int = 5) -> GridSearchCV:
    pipe = get_lr_pipeline()

    # Define parameter grid (note 'logreg__' prefix)
    param_grid = {
        'logreg__penalty': ['l1', 'l2'],  # 'elasticnet'
        'logreg__C': [0.01, 0.1, 1, 10],
        'logreg__solver': ['saga'],  # 'saga' supports all penalties
        # 'logreg__l1_ratio': [0, 0.5, 1]  # Only used with 'elasticnet'

    }
    f1 = make_scorer(f1_score)
    grid = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring=f1, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid


def get_dt_pipeline() -> Pipeline:
    clf = DecisionTreeClassifier(random_state=42)
    return clf


def create_dt_grid_search(X_train, y_train, cv: int = 5) -> GridSearchCV:
    # Define the classifier
    clf = get_dt_pipeline()

    # Define the parameter grid
    param_grid = {
        'criterion': ['gini', 'entropy', 'log_loss'],  # Available in newer versions of sklearn
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    f1 = make_scorer(f1_score)
    grid = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=f1, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid


##
def main() -> None:
    ## Load your dataset
    df = pd.read_csv('churn_training.csv').set_index('customerID')
    submission_df = pd.read_csv('churn_holdout.csv').set_index('customerID').drop(columns=['Churn'])
    df_pp = preprocess(df.copy())
    submission_df_pp = preprocess(submission_df, is_submission=True)
    assert df.shape[1] == submission_df.shape[
        1] + 1  # make sure both datasets have the same number of columns after preprocessing
    ##
    for col in df:
        print(df[col].value_counts(dropna=False))
        print('*' * 20)
    ##
    for col in df_pp:
        print('' * 10, col, '' * 10)
        dtype_per_col = df_pp[col].dtypes
        print(dtype_per_col)
        assert dtype_per_col == dtype('int64') or dtype_per_col == dtype('float64')
        if col != 'Churn':
            dtype_per_col_submission = submission_df_pp[col].dtypes
            assert dtype_per_col_submission == dtype('int64') or dtype_per_col_submission == dtype('float64')
    ## Show rows with missing values
    df[df.isna().sum(axis=1) == 1]  # Check for NaN values
    assert df_pp.isna().sum().sum() == 0
    assert submission_df_pp.isna().sum().sum() == 0
    ##
    X = df_pp.drop('Churn', axis=1)
    y = df_pp['Churn']
    ## Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    ##
    grid_per_clf = {
        'KNN': create_knn_grid_search(X_train, y_train),
        'LR': create_logistic_regression_grid_search(X_train, y_train),
        'DT': create_dt_grid_search(X_train, y_train),
    }
    ##
    lst = []
    for clf_name, grid in grid_per_clf.items():
        best_params, score, best_estimator = grid.best_params_, grid.best_score_, grid.best_estimator_
        # print("Best parameters:", best_params)
        # print("Best cross-validation score: {:.2f}".format(score))
        item = (clf_name, score, best_params, best_estimator)
        lst.append(item)
    df_results = pd.DataFrame(lst, columns=['Classifier', 'Score', 'Best Parameters', 'Best Estimator'])
    df_results = df_results.sort_values(by='Score', ascending=False)
    print(df_results)
    best_model_dct = df_results.iloc[0].to_dict()
    # best_model_str, best_model_prams = best_model_dct['Classifier'], best_model_dct['Best Parameters']
    best_estimator = best_model_dct['Best Estimator']
    ## Evaluate the best model on the test set
    y_pred = best_estimator.predict(X_test)
    f1_score = calc_f1(y_test, y_pred)  # 0.59
    print(f'Evaluation: F1 Score on Test Set: {f1_score:.2f}')
    ## Submission
    y_pred_submission = best_estimator.predict(submission_df_pp)
    to_submit = pd.DataFrame(np.array([submission_df_pp.index, y_pred_submission]).T, columns=['customerID', 'Churn'])
    to_submit.to_csv('YOUR_ID_HERE', index=False, header=False)
    ##


print("asd")
##
main()
