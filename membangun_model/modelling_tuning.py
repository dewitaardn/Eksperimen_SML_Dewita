import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

dagshub.init(
    repo_owner="dewitaardn",
    repo_name="Eksperimen_SML_Dewita",
    mlflow=True
)

DATA_DIR = "Membangun_model/heartDisease_preprocessing"
X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test  = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/Y_train.csv").values.ravel()
y_test  = pd.read_csv(f"{DATA_DIR}/Y_test.csv").values.ravel()

# Hyperparameter grid
hyparam_grid = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l2"],
    "solver": ["liblinear"]
}

base_model = LogisticRegression(max_iter=1000)

grid = GridSearchCV(
    estimator=base_model,
    param_grid=hyparam_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

with mlflow.start_run(run_name="tuned_logreg"):
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    # Log params
    for k, v in grid.best_params_.items():
        mlflow.log_param(k, v)

    # Log metrics 
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Log artifact
    mlflow.sklearn.log_model(best_model, "model")
    print("Best parameters:", grid.best_params_)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
