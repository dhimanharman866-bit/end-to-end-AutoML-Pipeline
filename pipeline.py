# pipeline.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def detect_problem_type(df, target):
    if df[target].nunique() < 10:
        return "Classification"
    else:
        return "Regression"


def build_preprocessing(X):
    numeric_cols = X.select_dtypes(exclude=['object']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, cat_cols)
    ])

    return preprocessor


def train_pipeline(file, target_column):

    df = pd.read_csv(file.file)

    if target_column not in df.columns:
        return {"error": "Target column not found"}

    problem_type = detect_problem_type(df, target_column)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessing(X)

    if problem_type == "Classification":
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier()
        }
    else:
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor()
        }

    best_score = -np.inf
    best_pipeline = None
    best_model_name = None

    for name, model in models.items():

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        if problem_type == "Classification":
            score = accuracy_score(y_test, y_pred)
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse=np.sqrt(mse)
            score=-mse

        if score > best_score:
            best_score = score
            best_pipeline = pipeline
            best_model_name = name

    joblib.dump(best_pipeline, "models/best_model.pkl")

    return {
        "problem_type": problem_type,
        "best_model": best_model_name,
        "score": float(best_score)
    }