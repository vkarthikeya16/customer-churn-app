# app.py — Streamlit Churn Prediction (Fully Corrected and Enhanced)

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")



# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("E Commerce Dataset.csv")

# Train model
@st.cache_resource
def train_balanced_model():
    df = load_data()
    df = df.drop(columns=['CustomerID']) if 'CustomerID' in df.columns else df
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    X_train, _, y_train, _ = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler, X.columns.tolist()

model, scaler, trained_feature_names = train_balanced_model()

st.title("📊 Customer Churn Prediction")
st.markdown("Use the form below to evaluate churn risk for a customer.")

# Upload CSV for batch prediction
st.sidebar.subheader("📁 Upload CSV for Batch Prediction")
batch_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if batch_file:
    df = pd.read_csv(batch_file)
    df = pd.get_dummies(df)
    for col in trained_feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[trained_feature_names]
    scaled = scaler.transform(df)
    preds = model.predict(scaled)
    probs = model.predict_proba(scaled)[:, 1]
    df_results = pd.DataFrame({
        "Churn Flag (0=No, 1=Yes)": preds,
        "Churn_Probability": probs,
        "Churns?": ["YES" if p == 1 else "NO" for p in preds],
        "Interpretation": ["⚠️ Likely to Churn" if p == 1 else "❌ Will Not Churn" for p in preds]
    })
    st.write("### 🔄 Batch Prediction Results")

    def highlight_churn(cell):
        if cell == "⚠️ Likely to Churn":
            return 'background-color: #ffcccc; color: black;'
        elif cell == "❌ Will Not Churn":
            return 'background-color: #ccffcc; color: black;'
        else:
            return ''

    st.dataframe(df_results.style.applymap(highlight_churn, subset=["Interpretation"]))

    # 🔢 Summary Statistics
    total = len(df_results)
    churned = df_results["Churn Flag (0=No, 1=Yes)"].sum()
    not_churned = total - churned
    churn_rate = churned / total * 100

    st.markdown("### 📊 Summary Stats")
    st.markdown(f"- Total customers predicted: **{total}**")
    st.markdown(f"- Customers likely to churn: **{churned}**")
    st.markdown(f"- Customers unlikely to churn: **{not_churned}**")
    st.markdown(f"- Churn Rate: **{churn_rate:.2f}%**")

    csv_out = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Results", csv_out, "predictions.csv", "text/csv")

# --- Prediction Form ---
with st.form("churn_form"):
    st.subheader("🧾 Customer Profile")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        tenure = st.number_input("Tenure (months)", 0, 100, 12)
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        warehouse_to_home = st.number_input("Warehouse to Home Distance", 0, 100, 10)
        number_of_devices = st.number_input("Devices Registered", 1, 10, 2)
        satisfaction_score = st.slider("Satisfaction Score", 1, 5, 3)
        number_of_address = st.number_input("Number of Addresses", 1, 10, 1)
    with col2:
        hour_spend_on_app = st.number_input("Hours Spent on App", 0.0, 24.0, 2.5)
        complain = st.selectbox("Complained Recently?", ["Yes", "No"])
        order_amount_hike = st.number_input("Order Amount Hike (%)", 0, 100, 20)
        coupon_used = st.number_input("Coupons Used", 0, 100, 3)
        order_count = st.number_input("Order Count", 0, 100, 5)
        day_since_last_order = st.number_input("Days Since Last Order", 0, 365, 12)
        cashback_amount = st.number_input("Cashback Amount", 0, 10000, 150)

    st.subheader("📦 Customer Preferences")
    col3, col4 = st.columns(2)
    with col3:
        preferred_login_device = st.selectbox("Login Device", ["Mobile Phone", "Phone"])
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col4:
        preferred_payment_mode = st.selectbox("Payment Mode", ["COD (Cash on Delivery)", "Credit Card", "Debit Card", "E wallet", "UPI"])
        order_category = st.selectbox("Preferred Order Category", ["Grocery", "Laptop & Accessory", "Mobile", "Mobile Phone", "Others"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])

    submitted = st.form_submit_button("🔍 Predict Churn")

if submitted:
    input_data = pd.DataFrame({
        'Tenure': [tenure],
        'CityTier': [city_tier],
        'WarehouseToHome': [warehouse_to_home],
        'HourSpendOnApp': [hour_spend_on_app],
        'NumberOfDeviceRegistered': [number_of_devices],
        'SatisfactionScore': [satisfaction_score],
        'NumberOfAddress': [number_of_address],
        'Complain': [1 if complain == 'Yes' else 0],
        'OrderAmountHikeFromlastYear': [order_amount_hike],
        'CouponUsed': [coupon_used],
        'OrderCount': [order_count],
        'DaySinceLastOrder': [day_since_last_order],
        'CashbackAmount': [cashback_amount],
        'PreferredLoginDevice_Mobile Phone': [1 if preferred_login_device == 'Mobile Phone' else 0],
        'PreferredLoginDevice_Phone': [1 if preferred_login_device == 'Phone' else 0],
        'PreferredPaymentMode_COD': [1 if preferred_payment_mode == 'COD (Cash on Delivery)' else 0],
        'PreferredPaymentMode_Cash on Delivery': [1 if preferred_payment_mode == 'COD (Cash on Delivery)' else 0],
        'PreferredPaymentMode_Credit Card': [1 if preferred_payment_mode == 'Credit Card' else 0],
        'PreferredPaymentMode_Debit Card': [1 if preferred_payment_mode == 'Debit Card' else 0],
        'PreferredPaymentMode_E wallet': [1 if preferred_payment_mode == 'E wallet' else 0],
        'PreferredPaymentMode_UPI': [1 if preferred_payment_mode == 'UPI' else 0],
        'Gender_Male': [1 if gender == 'Male' else 0],
        'PreferedOrderCat_Grocery': [1 if order_category == 'Grocery' else 0],
        'PreferedOrderCat_Laptop & Accessory': [1 if order_category == 'Laptop & Accessory' else 0],
        'PreferedOrderCat_Mobile': [1 if order_category == 'Mobile' else 0],
        'PreferedOrderCat_Mobile Phone': [1 if order_category == 'Mobile Phone' else 0],
        'PreferedOrderCat_Others': [1 if order_category == 'Others' else 0],
        'MaritalStatus_Married': [1 if marital_status == 'Married' else 0],
        'MaritalStatus_Single': [1 if marital_status == 'Single' else 0]
    })
    for col in trained_feature_names:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[trained_feature_names]
    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = int(prob > 0.5)

    st.markdown("### 📈 Churn Risk Gauge")
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 1]},
            'bar': {'color': "red" if prob > 0.7 else "orange" if prob > 0.4 else "green"},
            'steps' : [
                {'range': [0, 0.4], 'color': "lightgreen"},
                {'range': [0.4, 0.7], 'color': "orange"},
                {'range': [0.7, 1], 'color': "red"}
            ]
        },
        number = {"valueformat": ".2%"},
        title = {'text': "Churn Probability"}
    ))
    st.plotly_chart(fig, use_container_width=True)

    if prediction == 1:
        st.warning(f"⚠️ Prediction: Customer is likely to churn.\n🔸 Probability: {prob:.2f}")
    else:
        st.success(f"✅ Prediction: Customer is not likely to churn.\n🟢 Probability: {1 - prob:.2f}")
