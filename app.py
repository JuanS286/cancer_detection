import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model  # Or joblib if using an sklearn model
import pickle

# Load the trained model
model = load_model('cancermodel.h5')  # Update with your saved model path
scaler = pickle.load(open('scaler.pkl', 'rb'))
# Define the feature names and their ranges
features = {
    "mean radius": (6.981, 28.11),
    "mean perimeter": (43.79, 188.5),
    "mean area": (143.5, 2501.0),
    "mean compactness": (0.01938, 0.3454),
    "mean concavity": (0., 0.4268),
    "mean concave points": (0., 0.2012),
    "radius error": (0.1115, 2.873),
    "perimeter error": (0.757, 21.98),
    "area error": (6.802, 542.2),
    "worst radius": (7.93, 36.04),
    "worst perimeter": (50.41, 251.2),
    "worst area": (185.2, 4254.),
    "worst compactness": (0.02729, 1.058),
    "worst concavity": (0., 1.252),
    "worst concave points": (0., 0.291),
}

# Streamlit app layout
st.title("Breast Cancer Prediction")
st.write("Enter the values for the features to generate a prediction.")

# Collect user input for each feature
user_input = {}
for feature, (min_val, max_val) in features.items():
    user_input[feature] = st.number_input(
        f"{feature} ({min_val} to {max_val})",
        min_value=min_val,
        max_value=max_val,
        step=(max_val - min_val) / 100,
    )

# Create a DataFrame from user input
input_df = pd.DataFrame([user_input])

# Display the input data
st.subheader("Input Data")
st.write(input_df)

# Predict if button is pressed
if st.button("Generate Prediction"):
    df = scaler.transform(input_df)
    prediction = model.predict(df)
    predicted_class = "Malignant" if prediction[0][0] > 0.5 else "Benign"

    st.subheader("Prediction")
    st.write(f"The predicted class is: **{predicted_class}**")
    st.write(f"Prediction Confidence: **{prediction[0][0]:.2f}**")
