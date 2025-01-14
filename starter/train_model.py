# Script to train machine learning model.

from sklearn.model_selection import train_test_split
# Add the necessary imports for the starter code.
import pandas as pd
import numpy as np
from ml.data import process_data
from sklearn.linear_model import LogisticRegression
import joblib
from ml.model import compute_model_metrics, inference


def save_model():
    # Add code to load in the data.
    data = pd.read_csv("../data/census.csv")
    data.replace(r'^\s*\?\s*$', np.nan, regex=True, inplace=True)
    data.columns = data.columns.str.strip()
    '''Optional enhancement, use K-fold cross validation
    instead of a train-test split.'''
    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_feature=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_feature=cat_features, label="salary",
        encoder=encoder, lb=lb, training=False
    )

    # Train and save a model.
    model = LogisticRegression(max_iter=200)

    model.fit(X_train, y_train)

    joblib.dump(model, "../model/model.joblib")
    joblib.dump(encoder, "../model/encoder.joblib")
    joblib.dump(lb, "../model/lb.joblib")

    pred = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, pred)
    print(f" Precision: {precision:.3f}, "
          f"Recall: {recall:.3f}, fbeta: {fbeta:.3f}")

    return X_train, y_train, X_test, y_test, model


if __name__ == "__main__":
    save_model()
