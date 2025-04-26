
# app.py — Deployment Ready

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
trained_feature_names = joblib.load('feature_names.pkl')

st.title("📊 Customer Churn Prediction App")

st.sidebar.header("📁 Upload CSV for Batch Testing")
batch_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

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
        "Prediction": preds,
        "Churn_Probability": probs,
        "Churns?": ["YES" if p == 1 else "NO" for p in preds],
        "Interpretation": ["⚠️ Likely to Churn" if p == 1 else "✅ Will Not Churn" for p in preds]
    })

    def highlight_churn(cell):
        if cell == "⚠️ Likely to Churn":
            return 'background-color: #ffcccc; color: black;'
        elif cell == "✅ Will Not Churn":
            return 'background-color: #ccffcc; color: black;'
        else:
            return ''

    st.write("### 🔎 Batch Prediction Results")
    st.dataframe(df_results.style.applymap(highlight_churn, subset=["Interpretation"]))

    st.markdown("### 📊 Summary Stats")
    total = len(df_results)
    churned = df_results['Prediction'].sum()
    st.markdown(f"- Total Customers: **{total}**")
    st.markdown(f"- Customers Likely to Churn: **{churned}**")
    st.markdown(f"- Churn Rate: **{(churned/total)*100:.2f}%**")

    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Results", csv, "predictions.csv", "text/csv")

st.header("🧾 Predict Single Customer")

with st.form("churn_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        tenure = st.number_input("Tenure (months)", 0, 100, 12)
        city_tier = st.selectbox("City Tier", [1,2,3])
        warehouse_to_home = st.number_input("Warehouse to Home Distance", 0, 50, 10)
        devices = st.number_input("Devices Registered", 1, 10, 2)
        satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
        addresses = st.number_input("Number of Addresses", 1, 10, 1)
    with col2:
        hours_on_app = st.number_input("Hours Spent on App", 0.0, 24.0, 2.5)
        complain = st.selectbox("Complained?", ["Yes", "No"])
        order_hike = st.number_input("Order Amount Hike (%)", 0, 100, 20)
        coupons = st.number_input("Coupons Used", 0, 100, 5)
        order_count = st.number_input("Order Count", 0, 100, 10)
        days_since = st.number_input("Days Since Last Order", 0, 365, 20)
        cashback = st.number_input("Cashback Amount", 0, 10000, 200)

    st.subheader("🛒 Customer Preferences")
    col3, col4 = st.columns(2)
    with col3:
        login_device = st.selectbox("Login Device", ["Mobile Phone", "Phone"])
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col4:
        payment_mode = st.selectbox("Payment Mode", ["COD (Cash on Delivery)", "Credit Card", "Debit Card", "E wallet", "UPI"])
        category = st.selectbox("Preferred Order Category", ["Grocery", "Laptop & Accessory", "Mobile", "Mobile Phone", "Others"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])

    submit = st.form_submit_button("🔍 Predict Now")

if submit:
    input_data = pd.DataFrame({
        'Age': [age],
        'Tenure': [tenure],
        'CityTier': [city_tier],
        'WarehouseToHome': [warehouse_to_home],
        'HourSpendOnApp': [hours_on_app],
        'NumberOfDeviceRegistered': [devices],
        'SatisfactionScore': [satisfaction],
        'NumberOfAddress': [addresses],
        'Complain': [1 if complain == 'Yes' else 0],
        'OrderAmountHikeFromlastYear': [order_hike],
        'CouponUsed': [coupons],
        'OrderCount': [order_count],
        'DaySinceLastOrder': [days_since],
        'CashbackAmount': [cashback],
        'PreferredLoginDevice_Mobile Phone': [1 if login_device == 'Mobile Phone' else 0],
        'PreferredLoginDevice_Phone': [1 if login_device == 'Phone' else 0],
        'PreferredPaymentMode_COD': [1 if payment_mode == 'COD (Cash on Delivery)' else 0],
        'PreferredPaymentMode_Cash on Delivery': [1 if payment_mode == 'COD (Cash on Delivery)' else 0],
        'PreferredPaymentMode_Credit Card': [1 if payment_mode == 'Credit Card' else 0],
        'PreferredPaymentMode_Debit Card': [1 if payment_mode == 'Debit Card' else 0],
        'PreferredPaymentMode_E wallet': [1 if payment_mode == 'E wallet' else 0],
        'PreferredPaymentMode_UPI': [1 if payment_mode == 'UPI' else 0],
        'Gender_Male': [1 if gender == 'Male' else 0],
        'PreferedOrderCat_Grocery': [1 if category == 'Grocery' else 0],
        'PreferedOrderCat_Laptop & Accessory': [1 if category == 'Laptop & Accessory' else 0],
        'PreferedOrderCat_Mobile': [1 if category == 'Mobile' else 0],
        'PreferedOrderCat_Mobile Phone': [1 if category == 'Mobile Phone' else 0],
        'PreferedOrderCat_Others': [1 if category == 'Others' else 0],
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

    st.subheader("📈 Churn Risk Meter")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        gauge={
            'axis': {'range': [0,1]},
            'bar': {'color': "red" if prob > 0.7 else "orange" if prob > 0.4 else "green"},
            'steps': [
                {'range': [0,0.4], 'color': "lightgreen"},
                {'range': [0.4,0.7], 'color': "orange"},
                {'range': [0.7,1], 'color': "red"}
            ]
        },
        number={"valueformat": ".2%"},
        title={'text': "Churn Probability"}
    ))
    st.plotly_chart(fig, use_container_width=True)

    if prediction == 1:
        st.warning(f"⚠️ Customer is likely to churn. | 🔸 Probability: {prob:.2f}")
    else:
        st.success(f"✅ Customer is NOT likely to churn. | 🟢 Probability: {(1-prob):.2f}")

st.markdown("---")
st.markdown("Built with ❤️ by [Your Name Here]", unsafe_allow_html=True)
