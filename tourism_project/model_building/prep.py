"""
prep.py - Data preparation pipeline for Visit With Us Wellness Package Prediction MLOps Project

Steps:
1. Load dataset from Hugging Face Hub OR local fallback
2. Clean dataset (remove bad cols, fix inconsistencies)
3. Define numerical & categorical features
4. Build preprocessing pipeline
5. Transform data
6. Split into train/test
7. Save processed files locally
8. Save preprocessing pipeline (VERY IMPORTANT)
9. Upload processed data + preprocessor to Hugging Face

"""

# ========== IMPORTS ==========
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from huggingface_hub import HfApi


# ========== ENV VARS ==========
HF_DATASET_REPO_ID = "Quantum9999/Tourism-Package-Prediction"     # username/your-dataset-repo
HF_TOKEN = os.getenv("MLOPS_TOKEN")
RAW_DATA_FILE = "tourism.csv"

api = HfApi(token=HF_TOKEN)

# Ensure local data folder exists
os.makedirs("tourism_project/data", exist_ok=True)


# ============================================================
# STEP 1: LOAD DATASET (HuggingFace → local)
# ============================================================
def load_dataset():
    try:
        if HF_DATASET_REPO_ID:
            print("Downloading dataset from Hugging Face Hub...")
            api.hf_hub_download(
                repo_id=HF_DATASET_REPO_ID,
                filename=RAW_DATA_FILE,
                repo_type="dataset",
                local_dir="tourism_project/data"
            )
            df = pd.read_csv(f"tourism_project/data/{RAW_DATA_FILE}")
            print("Dataset loaded successfully from Hugging Face Hub.")
        else:
            df = pd.read_csv(f"tourism_project/data/{RAW_DATA_FILE}")
            print("Dataset loaded successfully from local path.")
        return df

    except Exception as e:
        print("ERROR loading dataset:", e)
        raise


df = load_dataset()


# ============================================================
# STEP 2: DATA CLEANING
# ============================================================

# Remove unwanted index column
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)
    print("Removed column: Unnamed: 0")

# Fix inconsistent Gender values
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    print("Fixed Gender inconsistencies.")

# Drop ID column
if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)


# ============================================================
# STEP 3: FEATURE DEFINITIONS
# ============================================================

target = "ProdTaken"

numerical_features = [
    "Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "NumberOfTrips", "NumberOfChildrenVisiting", "MonthlyIncome",
    "PitchSatisfactionScore", "NumberOfFollowups", "DurationOfPitch"
]

categorical_features = [
    "TypeofContact", "Occupation", "Gender", "MaritalStatus",
    "Passport", "OwnCar", "Designation", "ProductPitched"
]

# Prepare X, y
X = df.drop(columns=[target])
y = df[target]


# ============================================================
# STEP 4: BUILD PREPROCESSING PIPELINE
# ============================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore",sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

pipeline = Pipeline(steps=[("preprocessor", preprocessor)])


# ============================================================
# STEP 5: TRANSFORM DATA
# ============================================================

X_processed = pipeline.fit_transform(X)

# Get feature names after encoding
ohe_features = pipeline.named_steps["preprocessor"] \
                       .named_transformers_["cat"] \
                       .named_steps["encoder"] \
                       .get_feature_names_out(categorical_features)

processed_feature_names = numerical_features + list(ohe_features)

X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)


# ============================================================
# STEP 6: TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_processed_df, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# STEP 7: SAVE DATA LOCALLY
# ============================================================

X_train.to_csv("tourism_project/data/X_train.csv", index=False)
X_test.to_csv("tourism_project/data/X_test.csv", index=False)
y_train.to_csv("tourism_project/data/y_train.csv", index=False)
y_test.to_csv("tourism_project/data/y_test.csv", index=False)

print("Saved processed train/test datasets locally.")


# ============================================================
# STEP 8: SAVE PREPROCESSOR PIPELINE (IMPORTANT!)
# ============================================================

joblib.dump(pipeline, "tourism_project/data/preprocessing_pipeline.pkl")
print("preprocessing_pipeline.pkl saved.")


# ============================================================
# STEP 9: UPLOAD TO HUGGING FACE
# ============================================================

if HF_DATASET_REPO_ID:
    try:
        api.upload_folder(
            folder_path="tourism_project/data",
            repo_id=HF_DATASET_REPO_ID,
            repo_type="dataset"
        )
        print("Uploaded processed data + pipeline to Hugging Face Dataset Repo.")

    except Exception as e:
        print("Error uploading to Hugging Face:", e)

print("\n Data preparation complete! Your project is ready for model training.")
