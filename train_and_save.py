# train_and_save.py
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor

# 1. Load the data
df = pd.read_csv("./data/house_prices_srilanka.csv")

# 2. Define exact feature structures
categorical_cols = ['district', 'area', 'water_supply', 'electricity']
numerical_cols = ['perch', 'bedrooms', 'bathrooms', 'kitchen_area_sqft', 'parking_spots', 'floors', 'year_built']
boolean_cols = ['has_garden', 'has_ac']

X = df[numerical_cols + categorical_cols + boolean_cols]
y = df['price_lkr']

# 3. Build transformers
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

bool_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, numerical_cols),
    ('cat', cat_transformer, categorical_cols),
    ('bool', bool_transformer, boolean_cols)
])

# 4. Create and train pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(random_state=42))
])

model_pipeline.fit(X, y)

# 5. Save the trained artifact securely
with open("house_model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

print("✅ Success: 'house_model.pkl' created and saved successfully!")