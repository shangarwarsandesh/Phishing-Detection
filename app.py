import streamlit as st
import joblib
import pandas as pd
import numpy as np
from main import extract_features

# Load the trained model
model = joblib.load("phishing_model.pkl")

st.title("URL Phishing Detection")
st.write("Enter a URL below to check if it is legitimate or phishing.")

# User input
url = st.text_input("Enter a URL:")

if st.button("Check URL"):
    if url:
        try:
            # Extract features from the input URL
            features = extract_features(url)
            
            # Ensure the feature array matches the trained model input
            features_df = pd.DataFrame([features], columns=[
                "length_url", "length_hostname", "nb_dots", "nb_hyphens", "nb_at", "nb_qm",
                "nb_and", "nb_or", "nb_eq", "nb_underscore", "nb_tilde", "nb_percent",
                "nb_slash", "nb_star", "nb_colon", "nb_comma", "nb_semicolumn",
                "nb_dollar", "nb_space", "nb_www", "nb_com", "nb_dslash", "http_in_path",
                "https_token", "ratio_digits_url", "ratio_digits_host", "prefix_suffix",
                "random_domain", "shortening_service", "nb_subdomains", "phish_hints"
            ])
            
            # Make prediction
            prediction = model.predict(features_df)
            result = "Legitimate" if prediction[0] == "legitimate" else "Phishing"
            
            # Display result
            st.success(f"The URL is classified as: **{result}**")
        except Exception as e:
            st.error(f"Error processing URL: {e}")
    else:
        st.warning("Please enter a valid URL.")
