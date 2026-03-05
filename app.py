import streamlit as st
import requests

st.title("AutoML System")

backend_url = "http://127.0.0.1:8000"

st.header("Train Model")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
target_column = st.text_input("Enter Target Column Name")

if st.button("Train Model"):
    if uploaded_file and target_column:
        files = {"file": uploaded_file}
        params = {"target_column": target_column}

        response = requests.post(
            f"{backend_url}/train",
            files=files,
            params=params
        )

        st.write(response.json())
    else:
        st.warning("Please upload file and enter target column.")

st.header("Make Prediction")

prediction_input = st.text_area(
    "Enter feature values as JSON (example: {\"age\":30,\"salary\":50000})"
)

if st.button("Predict"):
    if prediction_input:
        response = requests.post(
            f"{backend_url}/predict",
            json=eval(prediction_input)
        )
        st.write(response.json())