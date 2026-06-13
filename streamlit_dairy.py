import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load model
model = pickle.load(open("yadav_pickle", "rb"))

# Load and prepare full training data (same as ML code)
df_full = pd.read_csv(
    'yadav dairy 2023-2024 (English).csv',
    encoding='utf-8',
    usecols=[
        'Buffalo', 'Present', 'Buffalo_Purchase_Date', 'Buffalo_Died',
        'Buffalo_Sold_Date', 'Buffalo_Sold', 'Fresh_Buffalo / Calved_Buffalo',
        'Reminder', 'Pregnancy', 'Test', 'In_Doubt', 'Empty_Days', 'Empty',
        'OK', 'Pregnancy_Month', 'पड़ा', 'Doctor_Name', 'Month', 'Medicine',
        'Calf_Born', 'Calf_Born Reminder', 'Dry_Period', 'Care', 'Calving',
        'Days_Until_Calving', 'Age_in_Months', 'Age_in_Days',
        'Milk_Production_Started', 'Milk_Production_per_Month_(Liters)',
        'Daily_Milk_Production_Morning_(Liters)',
        'Daily_Milk_Production_Evening(Liters)', 'Total_Milk',
        'Empty_Buffalo_Reminder',
    ],
)

# Apply same imputation as training
imputer = SimpleImputer(strategy='constant', fill_value=0)
df_full['Buffalo_Purchase_Date'] = imputer.fit_transform(df_full[['Buffalo_Purchase_Date']]).ravel()
df_full['Buffalo_Died'] = imputer.fit_transform(df_full[['Buffalo_Died']]).ravel()
df_full['Daily_Milk_Production_Evening(Liters)'] = imputer.fit_transform(df_full[['Daily_Milk_Production_Evening(Liters)']]).ravel()

# Impute missing values
# Numeric columns: fill with mean
numeric_cols = df_full.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    num_imputer = SimpleImputer(strategy='mean')
    df_full[numeric_cols] = pd.DataFrame(
        num_imputer.fit_transform(df_full[numeric_cols]),
        columns=numeric_cols,
        index=df_full.index,
    )

# Categorical columns used by model: fill with most frequent (mode)
cat_cols = ['Buffalo', 'Present', 'Pregnancy', 'Empty_Buffalo_Reminder']
cat_cols = [c for c in cat_cols if c in df_full.columns]
if cat_cols:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df_full[cat_cols] = pd.DataFrame(
        cat_imputer.fit_transform(df_full[cat_cols]),
        columns=cat_cols,
        index=df_full.index,
    )

# Apply label encoding on filled categorical columns
encoder_buffalo = LabelEncoder()
encoder_present = LabelEncoder()
encoder_pregnancy = LabelEncoder()
encoder_empty_reminder = LabelEncoder()

if 'Present' in df_full.columns:
    df_full['Present'] = encoder_present.fit_transform(df_full['Present'].astype(str)).astype(np.int64)
if 'Buffalo' in df_full.columns:
    df_full['Buffalo'] = encoder_buffalo.fit_transform(df_full['Buffalo'].astype(str)).astype(np.int64)
if 'Empty_Buffalo_Reminder' in df_full.columns:
    df_full['Empty_Buffalo_Reminder'] = encoder_empty_reminder.fit_transform(df_full['Empty_Buffalo_Reminder'].astype(str)).astype(np.int64)
if 'Pregnancy' in df_full.columns:
    df_full['Pregnancy'] = encoder_pregnancy.fit_transform(df_full['Pregnancy'].astype(str)).astype(np.int64)

# Prepare numeric feature columns and Streamlit UI
X_numeric = df_full.drop('Total_Milk', axis=1).select_dtypes(include=[np.number])

# Prefer model's fitted feature names when available to avoid mismatch errors
try:
    model_feature_names = list(model.feature_names_in_)
except Exception:
    try:
        # xgboost booster
        model_feature_names = list(model.get_booster().feature_names)
    except Exception:
        model_feature_names = X_numeric.columns.tolist()

# Ensure feature_columns is a list of strings (order matters for model)
feature_columns = [str(c) for c in model_feature_names]

# Compute basic evaluation metrics on available data (if possible)
model_r2 = None
model_mae = None
model_rmse = None
try:
    if 'Total_Milk' in df_full.columns and all(col in df_full.columns for col in feature_columns):
        X_eval = df_full[feature_columns].astype(float)
        y_eval = df_full['Total_Milk'].astype(float)
        preds_eval = model.predict(X_eval)
        model_r2 = float(r2_score(y_eval, preds_eval))
        model_mae = float(mean_absolute_error(y_eval, preds_eval))
        import math
        model_rmse = float(math.sqrt(mean_squared_error(y_eval, preds_eval)))
except Exception:
    # If evaluation fails, leave metrics as None
    model_r2 = None
    model_mae = None
    model_rmse = None

st.title('🐮 Buffalo Total Milk Production Prediction 🐮')
st.write("Buffalo Predictor")
st.write("Enter your buffalo details below and click Predict Total Milk to get your estimated milk yield.")

# Get unique categories from encoders (fitted above)
buffalo_categories = encoder_buffalo.classes_.tolist() if hasattr(encoder_buffalo, 'classes_') else []
present_categories = encoder_present.classes_.tolist() if hasattr(encoder_present, 'classes_') else []
pregnancy_categories = encoder_pregnancy.classes_.tolist() if hasattr(encoder_pregnancy, 'classes_') else []
empty_reminder_categories = encoder_empty_reminder.classes_.tolist() if hasattr(encoder_empty_reminder, 'classes_') else []

# User inputs with actual category names
Buffalo_type = st.selectbox("Buffalo Type:", options=buffalo_categories)
present = st.selectbox("Present Status:", options=present_categories)
buffalo_died = st.selectbox("Buffalo Died:", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
pregnancy = st.selectbox("Pregnancy Status:", options=pregnancy_categories)
Empty_Buffalo_Reminder = st.selectbox("Empty Buffalo Reminder:", options=empty_reminder_categories)
purchase_date_encoded = st.number_input("Buffalo Purchase Date (Encoded Number):", value=0.0)
daily_milk_evening = st.number_input("Daily Milk Producing Evening (Liters):", min_value=0.0, max_value=50.0, value=5.0)

if st.button("Predict Total Milk"):
    # Transform categorical inputs using loaded encoders
    buffalo_encoded = int(encoder_buffalo.transform(np.array([Buffalo_type], dtype=object))[0])
    present_encoded = int(encoder_present.transform(np.array([present], dtype=object))[0])
    pregnancy_encoded = int(encoder_pregnancy.transform(np.array([pregnancy], dtype=object))[0])
    reminder_encoded = int(encoder_empty_reminder.transform(np.array([Empty_Buffalo_Reminder], dtype=object))[0])
    
    # Create a dataframe with only the feature names expected by the model
    input_data = pd.DataFrame({col: [0.0] for col in feature_columns})

    # Fill in the values that user provided, but only for features the model expects
    if 'Buffalo' in input_data.columns:
        input_data.at[0, 'Buffalo'] = buffalo_encoded
    if 'Present' in input_data.columns:
        input_data.at[0, 'Present'] = present_encoded
    if 'Buffalo_Purchase_Date' in input_data.columns:
        input_data.at[0, 'Buffalo_Purchase_Date'] = purchase_date_encoded
    if 'Buffalo_Died' in input_data.columns:
        input_data.at[0, 'Buffalo_Died'] = buffalo_died
    if 'Pregnancy' in input_data.columns:
        input_data.at[0, 'Pregnancy'] = pregnancy_encoded
    if 'Daily_Milk_Production_Evening(Liters)' in input_data.columns:
        input_data.at[0, 'Daily_Milk_Production_Evening(Liters)'] = daily_milk_evening
    if 'Empty_Buffalo_Reminder' in input_data.columns:
        input_data.at[0, 'Empty_Buffalo_Reminder'] = reminder_encoded
    
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Total Milk: {prediction:.2f} Liters")
        # Show model evaluation metrics (if computed)
        if model_r2 is not None:
            st.write(f"Model R²: {model_r2:.3f} — MAE: {model_mae:.2f} — RMSE: {model_rmse:.2f}")
        else:
            st.write("Model evaluation metrics: N/A")
        st.write("Yeh prediction aapke diye gaye inputs ke hisaab se banayi gayi hai.")
        st.write("Note: model prediction ke liye baaki fields default values se set kiye gaye hain.")

        feature_data = pd.DataFrame({
            "feature": input_data.columns,
            "value": input_data.iloc[0].values,
        }).set_index("feature")

        st.subheader("Input Feature Breakdown")
        st.bar_chart(feature_data)
        st.snow()
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.snow()
    
    




