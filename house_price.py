import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("data.csv")
print("Original shape:", df.shape)

# -----------------------------
# DATA CLEANING
# -----------------------------
df = df.drop_duplicates()
df = df.dropna()

# Remove extreme outliers in price
Q1 = df['price'].quantile(0.01)
Q99 = df['price'].quantile(0.99)
df = df[(df['price'] >= Q1) & (df['price'] <= Q99)]
print("After cleaning and outlier removal:", df.shape)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
# Convert date to numeric (days since earliest date)
df['date'] = pd.to_datetime(df['date'])
df['date_numeric'] = (df['date'] - df['date'].min()).dt.days

# Encode categorical columns using LabelEncoder
cat_cols = ['street', 'city', 'statezip', 'country']

# ---- 1. City encoder (needed for prediction) ----
city_encoder = LabelEncoder()
df['city'] = city_encoder.fit_transform(df['city'])
joblib.dump(city_encoder, "city_encoder.pkl")


# ---- 2. Other categorical columns ----
for col in ['street','statezip','country']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
   

# -----------------------------
# FEATURES & TARGET
# -----------------------------
# Use numeric + label-encoded categorical columns
features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors',
            'waterfront','view','condition','sqft_above','sqft_basement',
            'yr_built','yr_renovated','date_numeric',
            'street','city','statezip','country']   # keep encoded columns

X = df[features]
y = df['price']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# RANDOM FOREST WITH HYPERPARAMETER TUNING
# -----------------------------
# Hyperparameter grid
param_grid = {
    'n_estimators': [200, 500, 800],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}


# RandomizedSearchCV
rf = RandomForestRegressor(
    n_estimators=300, 
    max_depth=20, 
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1,   # <-- MULTI-CORE SPEEDUP
    random_state=42
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Optimized Random Forest Results:")
print("RMSE:", rmse)
print("R2 Score:", r2)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False).head(20).plot(kind='bar', figsize=(15,6))
plt.title("Top 20 Feature Importances")

# -----------------------------
# SAVE TRAINED MODEL
# -----------------------------
joblib.dump(rf, "house_price_model.pkl")
print("Model saved successfully!")
plt.show()



