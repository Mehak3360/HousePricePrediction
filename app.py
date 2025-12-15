import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction App")
st.markdown("Enter the details of the house to predict its price.")

# -----------------------------
# Load model & encoder
# -----------------------------
model = joblib.load("house_price_model.pkl")
city_encoder = joblib.load("city_encoder.pkl")

# City options for dropdown
city_classes = list(city_encoder.classes_)

# -----------------------------
# Sidebar / Input widgets
# -----------------------------
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=1.0)
sqft_living = st.number_input("Living Area (sqft)", min_value=100, max_value=10000, value=1800)
sqft_lot = st.number_input("Lot Size (sqft)", min_value=100, max_value=50000, value=5000)
floors = st.number_input("Floors", min_value=1, max_value=5, value=2)
waterfront = st.selectbox("Waterfront", [0,1])
view = st.selectbox("View", [0,1,2,3,4])
condition = st.selectbox("Condition (1-5)", [1,2,3,4,5])
sqft_above = st.number_input("Above Ground Area (sqft)", min_value=100, max_value=10000, value=1600)
sqft_basement = st.number_input("Basement Area (sqft)", min_value=0, max_value=5000, value=200)
yr_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=1995)
yr_renovated = st.number_input("Year Renovated", min_value=0, max_value=2025, value=0)
date_numeric = st.number_input("Days Since Earliest Sale", min_value=0, max_value=20000, value=4000)

# Placeholders for categorical numeric columns
street = 0
statezip = 0
country = 0

# City dropdown
city_input = st.selectbox("City", city_classes)
city_encoded = city_encoder.transform([city_input])[0]

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict Price"):
    # Create input DataFrame
    input_df = pd.DataFrame([[
        bedrooms, bathrooms, sqft_living, sqft_lot, floors,
        waterfront, view, condition, sqft_above, sqft_basement,
        yr_built, yr_renovated, date_numeric,
        street, city_encoded, statezip, country
    ]], columns=['bedrooms','bathrooms','sqft_living','sqft_lot','floors',
                 'waterfront','view','condition','sqft_above','sqft_basement',
                 'yr_built','yr_renovated','date_numeric',
                 'street','city','statezip','country'])

    # Predict
    predicted_price = model.predict(input_df)[0]

    st.success(f"💰 Predicted House Price: ${predicted_price:,.2f}")

    # -----------------------------
    # Feature Importance Plot
    # -----------------------------
    st.subheader("Top 5 Feature Importances")
    feature_importances = pd.Series(model.feature_importances_, index=input_df.columns)
    top_features = feature_importances.sort_values(ascending=False).head(5)

    fig, ax = plt.subplots()
    top_features.plot(kind='bar', ax=ax)
    ax.set_ylabel("Importance")
    ax.set_title("Top 5 Features")
    st.pyplot(fig)
