import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download


# 1. Load Model from Hugging Face Model Hub
# ============================================================

MODEL_REPO = "Quantum9999/Tourism-Package-Prediction"
MODEL_FILENAME = "xgb_model.pkl"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, repo_type="model")
    model = joblib.load(model_path)
    return model

model = load_model()


# 2. Streamlit UI
# ============================================================

st.title(" Wellness Tourism Package Purchase Prediction")
st.write("Fill in the customer details below to predict whether they will purchase the new Wellness Tourism Package.")

st.markdown("---")

# 3. User Inputs
# ============================================================

def user_input_form():
    Age = st.number_input("Age", min_value=1, max_value=100, value=30)
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    NumberOfPersonVisiting = st.number_input("Number of People Visiting", min_value=1, max_value=10, value=1)
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
    NumberOfTrips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=2)
    NumberOfChildrenVisiting = st.number_input("Children Visiting (Under 5 Years)", min_value=0, max_value=5, value=0)
    MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=1000000, value=30000)
    PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=1, max_value=10, value=2)
    DurationOfPitch = st.number_input("Duration of Pitch (Minutes)", min_value=1, max_value=60, value=15)

    TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    Occupation = st.selectbox("Occupation", ["Salaried", "Self Employed", "Small Business", "Large Business", "Free Lancer"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
    Passport = st.selectbox("Passport", [0, 1])
    OwnCar = st.selectbox("Owns Car?", [0, 1])
    Designation = st.selectbox("Designation", ["Junior", "Senior", "Manager", "Executive", "Other"])
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

    # Create DataFrame
    data = pd.DataFrame({
        "Age": [Age],
        "CityTier": [CityTier],
        "NumberOfPersonVisiting": [NumberOfPersonVisiting],
        "PreferredPropertyStar": [PreferredPropertyStar],
        "NumberOfTrips": [NumberOfTrips],
        "NumberOfChildrenVisiting": [NumberOfChildrenVisiting],
        "MonthlyIncome": [MonthlyIncome],
        "PitchSatisfactionScore": [PitchSatisfactionScore],
        "NumberOfFollowups": [NumberOfFollowups],
        "DurationOfPitch": [DurationOfPitch],
        "TypeofContact": [TypeofContact],
        "Occupation": [Occupation],
        "Gender": [Gender],
        "MaritalStatus": [MaritalStatus],
        "Passport": [Passport],
        "OwnCar": [OwnCar],
        "Designation": [Designation],
        "ProductPitched": [ProductPitched]
    })

    return data

user_data = user_input_form()

st.markdown("---")


# 4. Preprocess User Input → MATCH Training Preprocessing
# ============================================================

# Categorical + numerical split (same as prep.py)
numerical_features = [
    'Age', 'CityTier', 'NumberOfPersonVisiting', 'PreferredPropertyStar',
    'NumberOfTrips', 'NumberOfChildrenVisiting', 'MonthlyIncome',
    'PitchSatisfactionScore', 'NumberOfFollowups', 'DurationOfPitch'
]

categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'MaritalStatus',
    'Passport', 'OwnCar', 'Designation', 'ProductPitched'
]

# Load preprocessors (generated in prep.py)
preprocessor_path = hf_hub_download(repo_id=MODEL_REPO, filename="preprocessing_pipeline.pkl", repo_type="model")
preprocessor = joblib.load(preprocessor_path)

processed_user_data = preprocessor.transform(user_data)


# 5. Make Prediction
# ============================================================

if st.button("Predict"):
    prediction = model.predict(processed_user_data)[0]
    proba = model.predict_proba(processed_user_data)[0][1]

    st.subheader(" Prediction Result")

    if prediction == 1:
        st.success(f" Customer is LIKELY to purchase the Wellness Tourism Package! (Confidence: {proba:.2f})")
    else:
        st.error(f" Customer is NOT likely to purchase the package. (Confidence: {proba:.2f})")

