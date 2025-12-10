import pandas as pd
import joblib
import numpy as np

# Load model & city encoder
model = joblib.load("house_price_model.pkl")
city_encoder = joblib.load("city_encoder.pkl")  # only city saved

# Input data
input_data = {
    "bedrooms": 3,
    "bathrooms": 2,
    "sqft_living": 1800,
    "sqft_lot": 5000,
    "floors": 2,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "sqft_above": 1600,
    "sqft_basement": 200,
    "yr_built": 1995,
    "yr_renovated": 0,
    "date_numeric": 4000,
    "street": 0,      # numeric placeholder
    "city": "Seattle",
    "statezip": 0,    # numeric placeholder
    "country": 0      # numeric placeholder
}

df = pd.DataFrame([input_data])

# Encode city safely
if df['city'][0] not in city_encoder.classes_:
    print("⚠ Warning: City not seen in training. Encoding as 'Unknown'.")
    city_encoder.classes_ = np.append(city_encoder.classes_, 'Unknown')
    df['city'] = ['Unknown']

df['city'] = city_encoder.transform(df['city'])

# Feature order same as training
features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors',
            'waterfront','view','condition','sqft_above','sqft_basement',
            'yr_built','yr_renovated','date_numeric',
            'street','city','statezip','country']

df = df[features]

# Predict
prediction = model.predict(df)[0]
print("\nPredicted House Price:", prediction)
