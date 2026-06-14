import utils.http_session
import utils.transport
import utils.session
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline

# Set page layout configuration
st.set_page_config(
    page_title="Sri Lanka House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ------------------------------------------------------------------
# 1. HELPER DATA MAPPINGS (Derived from house_prices_srilanka.csv)
# ------------------------------------------------------------------

# Comprehensive list of districts and corresponding available areas
DISTRICT_AREA_MAP = {
    "Polonnaruwa": ["Polonnaruwa Central"],
    "Matale": ["Matale Central"],
    "Mullaitivu": ["Mullaitivu Central"],
    "Anuradhapura": ["New Town", "Nuwaragam Palatha", "Madawachchiya"],
    "Batticaloa": ["Batticaloa Town", "Kallady", "Eravur"],
    "Kalutara": ["Beruwala", "Panadura", "Wadduwa", "Kalutara North"],
    "Ratnapura": ["Kuruwita", "Ratnapura Town", "Pelmadulla"],
    "Jaffna": ["Chunnakam", "Jaffna Town", "Kokuvil", "Nallur"],
    "Badulla": ["Badulla Town", "Bandarawela", "Hali Ela"],
    "Vavuniya": ["Vavuniya Central"],
    "Mannar": ["Mannar Central"],
    "Kilinochchi": ["Kilinochchi Central"],
    "Hambantota": ["Hambantota Town", "Tangalle", "Ambalantota"],
    "Puttalam": ["Puttalam Central"],
    "Kandy": ["Kandy City", "Peradeniya", "Katugastota", "Gatambe", "Tennekumbura"],
    "Gampaha": ["Ja-Ela", "Ragama", "Wattala", "Gampaha Town", "Kadawatha", "Negombo"],
    "Ampara": ["Ampara Central"],
    "Trincomalee": ["Nilaveli", "Uppuveli", "China Bay"],
    "Colombo": ["Narahenpita", "Nugegoda", "Wellawatte", "Borella", "Bambalapitiya", "Rajagiriya", "Mount Lavinia", "Kollupitiya", "Dehiwala"],
    "Matara": ["Matara Town", "Weligama", "Nupe", "Akurugoda"],
    "Kegalle": ["Kegalle Central"]
}

# ------------------------------------------------------------------
# 2. MODEL LOADING PLACEHOLDER / IN-LINE PIPELINE RE-BUILDER
# ------------------------------------------------------------------
@st.cache_resource
def load_predictive_model():
    """
    Since Streamlit runs independently, we recreate the scikit-learn Pipeline
    resembling your house_price_prediction.ipynb sequence.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import HistGradientBoostingRegressor
    
    # 1. Define standard features based on your dataset columns
    categorical_cols = ['district', 'area', 'water_supply', 'electricity']
    numerical_cols = ['perch', 'bedrooms', 'bathrooms', 'kitchen_area_sqft', 'parking_spots', 'floors', 'year_built']
    boolean_cols = ['has_garden', 'has_ac']
    
    # 2. Build preprocessing pipelines
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    bool_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_cols),
            ('cat', cat_transformer, categorical_cols),
            ('bool', bool_transformer, boolean_cols)
        ])
    
    # 3. Complete pipeline using your HistGradientBoostingRegressor parameters
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', HistGradientBoostingRegressor(random_state=42))
    ])
    
    # 4. Fit the model dynamically if the dataset is present locally
    csv_filename = "house_prices_srilanka.csv"
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        X = df[numerical_cols + categorical_cols + boolean_cols]
        y = df['price_lkr']
        model_pipeline.fit(X, y)
        return model_pipeline, True
    else:
        # Fallback dummy fitting so app interface operates without breaking errors
        # if the dataset isn't immediately detected on launch.
        dummy_data = pd.DataFrame([{
            'district': 'Colombo', 'area': 'Nugegoda', 'perch': 10, 'bedrooms': 3,
            'bathrooms': 2, 'kitchen_area_sqft': 120, 'parking_spots': 1,
            'has_garden': True, 'has_ac': False, 'water_supply': 'Both',
            'electricity': 'Three phase', 'floors': 2, 'year_built': 2020
        }])
        dummy_target = pd.Series([15000000.0])
        model_pipeline.fit(dummy_data, dummy_target)
        return model_pipeline, False

model, is_trained_on_real_data = load_predictive_model()

# ------------------------------------------------------------------
# 3. STREAMLIT USER INTERFACE DESIGN
# ------------------------------------------------------------------

st.title("🏠 Sri Lanka House Price Prediction Platform")
st.markdown("Enter property structural parameters below to securely predict expected real-estate valuation in **Sri Lankan Rupees (LKR)**.")

if not is_trained_on_real_data:
    st.warning("⚠️ `house_prices_srilanka.csv` was not found in the root app directory. The app is running in demonstration mode with mock weights.")
else:
    st.success("💪 Predictive model pipeline compiled successfully utilizing your uploaded dataset entries!")

# Structuring user inputs using columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📍 Location Specifications")
    selected_district = st.selectbox("Select District", options=sorted(list(DISTRICT_AREA_MAP.keys())))
    
    # Dynamically alternate area list based on selected district
    available_areas = DISTRICT_AREA_MAP[selected_district]
    selected_area = st.selectbox("Select Area / Town", options=sorted(available_areas))
    
    year_built = st.number_input("Year Built", min_value=1950, max_value=2026, value=2015, step=1)
    floors = st.slider("Number of Floors", min_value=1, max_value=5, value=1, step=1)

with col2:
    st.subheader("🏗️ Space & Layout Metrics")
    perch = st.number_input("Total Land Extent (Perches)", min_value=1.0, max_value=500.0, value=10.0, step=0.5)
    kitchen_sqft = st.number_input("Kitchen Area (SqFt)", min_value=10, max_value=2000, value=120, step=10)
    bedrooms = st.number_input("Bedrooms Count", min_value=1, max_value=20, value=3, step=1)
    bathrooms = st.number_input("Bathrooms Count", min_value=1, max_value=15, value=2, step=1)
    parking_spots = st.number_input("Vehicle Parking Spots Available", min_value=0, max_value=10, value=1, step=1)

with col3:
    st.subheader("⚡ Utility & Amenities")
    water_supply = st.selectbox("Water Supply Type", options=["Pipe-borne", "Well", "Both"])
    electricity = st.selectbox("Electricity Connection Type", options=["Single phase", "Three phase"])
    
    st.write("") # Padding spacing
    has_garden = st.checkbox("Property Features a Private Garden", value=False)
    has_ac = st.checkbox("Equipped with Air Conditioning (AC)", value=False)

# ------------------------------------------------------------------
# 4. PREDICTION INFERENCE TRIGGER EXECUTION
# ------------------------------------------------------------------
st.markdown("---")
if st.button("🔮 Calculate Estimated House Price", type="primary", use_container_width=True):
    
    # Process user input into single-row pandas dataframe format matching the scikit model exactly
    input_payload = pd.DataFrame([{
        "district": selected_district,
        "area": selected_area,
        "perch": int(perch),
        "bedrooms": int(bedrooms),
        "bathrooms": int(bathrooms),
        "kitchen_area_sqft": int(kitchen_sqft),
        "parking_spots": int(parking_spots),
        "has_garden": bool(has_garden),
        "has_ac": bool(has_ac),
        "water_supply": water_supply,
        "electricity": electricity,
        "floors": int(floors),
        "year_built": int(year_built)
    }])
    
    try:
        # Execute model pipeline scoring logic
        predicted_price = model.predict(input_payload)[0]
        
        # Guard clause against any anomalous negative calculations
        if predicted_price < 0:
            predicted_price = 0.0

        # Output the formatted valuation summary metric cards safely 
        st.subheader("📊 Model Inferred Estimation Results:")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric(
                label="Estimated Market Value (LKR)", 
                value=f"Rs. {predicted_price:,.2f}"
            )
        with res_col2:
            st.metric(
                label="Estimated Value (Millions LKR)", 
                value=f"Rs. {predicted_price / 1_000_000:.2f} Mn"
            )
            
    except Exception as prediction_error:
        st.error(f"Failed to generate evaluation projection: {prediction_error}")