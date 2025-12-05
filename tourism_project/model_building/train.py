import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, f1_score, accuracy_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import joblib
import os
from huggingface_hub import HfApi, login, hf_hub_download

# 1. Load Hugging Face Token (from environment variable) and Define Repos
# ============================================================

HF_TOKEN = os.getenv("MLOPS_TOKEN")     # <-- MUST be set in GitHub Secrets or notebook
HF_DATASET_REPO_ID = "Quantum9999/Tourism-Package-Prediction" # Repo where processed data is stored
HF_MODEL_REPO = "Quantum9999/Tourism-Package-Prediction"   # Repo where the trained model will be stored

if HF_TOKEN is None:
    raise ValueError(" ERROR: MLOPS_TOKEN is not set in environment variables!")

# Login to Hugging Face
login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)

# Ensure local data folder exists for processed data
os.makedirs("tourism_project/data", exist_ok=True)

# 2. Download Processed Data from Hugging Face Hub
# ============================================================
print("Downloading processed datasets from Hugging Face Hub...")
processed_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv", "preprocessing_pipeline.pkl"]
for filename in processed_files:
    try:
        hf_hub_download(
            repo_id=HF_DATASET_REPO_ID,
            filename=filename,
            repo_type="dataset", 
            local_dir="tourism_project/data"
        )
        print(f"Downloaded {filename} to tourism_project/data/")
    except Exception as e:
        print(f"Error downloading {filename} from Hugging Face Hub: {e}")
        raise # Re-raise the exception to fail the job if download fails


# 3. Load Processed Data (now available locally)
# ============================================================

print(" Loading processed datasets locally...")

X_train = pd.read_csv("tourism_project/data/X_train.csv")
X_test  = pd.read_csv("tourism_project/data/X_test.csv")
y_train = pd.read_csv("tourism_project/data/y_train.csv")["ProdTaken"]
y_test  = pd.read_csv("tourism_project/data/y_test.csv")["ProdTaken"]

print(" Data loaded successfully!")


# 4. Handle Class Imbalance
# ============================================================

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

print(f" Class imbalance ratio: {scale_pos_weight}")


# 5. MLflow Configuration
# ============================================================

mlflow.set_experiment("Tourism_Package_ProdTaken")


# 6. Define XGBoost Model + Hyperparameter Grid
# ============================================================

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

param_grid = {
    "n_estimators": [150, 200, 250],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="recall",   # PRIORITY metric
    cv=5,
    n_jobs=-1,
    verbose=1
)


# 7. TRAINING + MLflow Logging
# ============================================================

print(" Training started...")

with mlflow.start_run():

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print(" Best Parameters:", grid.best_params_)
    mlflow.log_params(grid.best_params_)

    # Predictions
    y_pred = best_model.predict(X_test)

    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("\n Model Performance:")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print(f"Accuracy: {acc}")

    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("accuracy", acc)


    # 8. Save Model Locally
    # ========================================================
    os.makedirs("tourism_project/final_model", exist_ok=True)
    model_path = "tourism_project/final_model/xgb_model.pkl"

    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path)

    print(f" Model saved: {model_path}")


    # 9. Upload Model to Hugging Face Model Hub
    # ========================================================
    print(f" Uploading model to HF Repo: {HF_MODEL_REPO}")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="xgb_model.pkl",
        repo_id=HF_MODEL_REPO,
        repo_type="model"
    )

print(" Training Completed & Model Uploaded to Hugging Face!")
