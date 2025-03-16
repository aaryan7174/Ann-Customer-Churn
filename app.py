import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_encoders():
    try:
        label_encoder_gender = pickle.load(open("label_encoder_gender.pkl", "rb"))
        onehot_encoder_geo = pickle.load(open("onehot_encoder_geo.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return label_encoder_gender, onehot_encoder_geo, scaler
    except Exception as e:
        st.error(f"Error loading encoders: {e}")
        return None, None, None

model = load_model()
label_encoder_gender, onehot_encoder_geo, scaler = load_encoders()

st.title("Customer Churn Prediction")
st.image("https://cdn.dribbble.com/userupload/18414675/file/original-701962c68d64f6d6209c1c9bd5993622.jpg?resize=752x&vertical=center", use_container_width=True)

# User Inputs
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92, 35)
balance = st.number_input("Balance", min_value=0.0, step=100.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=1000.0)
tenure = st.slider("Tenure", 0, 10, 5)
num_of_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"])
is_active_member = st.radio("Is Active Member?", ["Yes", "No"])

if model and label_encoder_gender and onehot_encoder_geo and scaler:
    # Prepare the input data
    gender_encoded = label_encoder_gender.transform([gender])[0]
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    
    input_data = np.hstack([geo_encoded, [[credit_score, gender_encoded, age, tenure, balance, num_of_products, has_cr_card == "Yes", is_active_member == "Yes", estimated_salary]]])
    input_data_scaled = scaler.transform(input_data)
    
    # Predict churn
    if st.button("Predict Churn"):
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]
        
        if prediction_proba > 0.5:
            st.error(f"The customer is likely to churn. (Probability: {prediction_proba:.2f})")
        else:
            st.success(f"The customer is unlikely to churn. (Probability: {prediction_proba:.2f})")
else:
    st.error("Encoders or model not loaded properly. Check your model files.")
