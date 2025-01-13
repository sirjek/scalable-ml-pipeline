from sklearn.metrics import precision_score, recall_score, fbeta_score
from train_model import save_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.compose import ColumnTransformer


def evaluate_slices(model, cat_feature):
    results = []
    data = pd.read_csv("../data/census.csv")
    data.replace(r'^\s*\?\s*$', np.nan, regex=True, inplace=True)
    data.columns = data.columns.str.strip()
    X = data.drop(columns=["salary"], axis=1)
    y = data["salary"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feature)
        ],
        remainder="passthrough"
    )

    X_encoded = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()
    X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=feature_names)
    lb = LabelBinarizer()
    y = lb.fit_transform(y).flatten()

    for feature in cat_feature:
        print(f"\nEvaluating slices for feature: {feature}")
        unique_values = X[feature].dropna().unique()
        for value in unique_values:
            slice_indices = X[feature] == value
            X_slice = X_encoded_df[slice_indices]
            y_slice = y[slice_indices]
            if len(y_slice) == 0:
                continue
            preds = model.predict(X_slice)
            precision = precision_score(y_slice, preds, zero_division=1)
            recall = recall_score(y_slice, preds, zero_division=1)
            f1 = fbeta_score(y_slice, preds, beta=1, zero_division=1)
            results.append({
                "Feature": feature,
                "Value": value,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
            })
            print(f"  Value: {value} - Precision: {precision:.3f}, "
                  f"Recall: {recall:.3f}, F1 Score: {f1:.3f}")
    return results


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, model = save_model()
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
    results = evaluate_slices(model, cat_features)

    with open("slice_output.txt", "w") as file:
        for item in results:
            file.write(f"{item}\n")
