import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(
    input_path="heart_disease_raw/heart_disease_uci.csv",
    output_dir="preprocessing/heartDisease_preprocessing"
):
    df = pd.read_csv(input_path)

    df = df.drop(columns=["id", "dataset"])

    df["target"] = (df["num"] > 0).astype(int)

    X = df.drop(columns=["num", "target"])
    y = df["target"]

    # SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_imputer = SimpleImputer(strategy="median")
    X_train_num = num_imputer.fit_transform(X_train[num_cols])
    X_test_num = num_imputer.transform(X_test[num_cols])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train_cat = cat_imputer.fit_transform(X_train[cat_cols])
    X_test_cat = cat_imputer.transform(X_test[cat_cols])


    encoder = OneHotEncoder(
        handle_unknown="ignore", 
        sparse_output=False,
        )
    X_train_cat_enc = encoder.fit_transform(X_train_cat)
    X_test_cat_enc = encoder.transform(X_test_cat)

    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    # GABUNGKAN 
    X_train_final = np.hstack([X_train_num_scaled, X_train_cat_enc])
    X_test_final = np.hstack([X_test_num_scaled, X_test_cat_enc])

    # SAVE OUTPUT
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(X_train_final).to_csv(f"{output_dir}/X_train.csv", index=False)
    pd.DataFrame(X_test_final).to_csv(f"{output_dir}/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv(f"{output_dir}/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv(f"{output_dir}/y_test.csv", index=False)

if __name__ == "__main__":
    preprocess_data()
