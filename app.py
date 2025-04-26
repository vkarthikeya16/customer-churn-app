# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Churn Prediction App", layout="centered")

# Load model, scaler, feature names
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
trained_feature_names = joblib.load("feature_names.pkl")

# Preprocessing function
def preprocess(df):
    df = pd.get_dummies(df, columns=[
        "Gender", "MaritalStatus", "PreferedOrderCat", "PreferredLoginDevice", "PreferredPaymentMode"
    ])
    for col in trained_feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[trained_feature_names]
    return df

st.title("📊 Customer Churn Prediction")

# Batch Prediction
st.sidebar.header("📁 Upload CSV for Batch Prediction")
batch_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if batch_file:
    df = pd.read_csv(batch_file)
    df = preprocess(df)
    
    scaled = scaler.transform(df)
    preds = model.predict(scaled)
    probs = model.predict_proba(scaled)[:, 1]

    df_results = pd.DataFrame({
        "Prediction (0=No,1=Yes)": preds,
        "Churn_Probability": probs,
        "Churn?": ["YES" if p == 1 else "NO" for p in preds],
        "Interpretation": ["⚠️ Likely to Churn" if p == 1 else "✅ Will Not Churn" for p in preds]
    })

    st.write("### 🔎 Batch Prediction Results")
    st.dataframe(df_results)

    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predictions", csv, "churn_predictions.csv", "text/csv")

st.header("🧑 Single Customer Prediction")

# Single Customer Input
with st.form("single_customer_form"):
    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", 18, 100, 30)
        Tenure = st.number_input("Tenure", 0, 100, 12)
        CityTier = st.selectbox("City Tier", [1, 2, 3])
        WarehouseToHome = st.number_input("Warehouse to Home Distance", 0, 50, 10)
        HourSpendOnApp = st.number_input("Hours Spend on App", 0.0, 24.0, 2.5)
        NumberOfDeviceRegistered = st.number_input("Devices Registered", 1, 10, 2)
        SatisfactionScore = st.slider("Satisfaction Score", 1, 5, 3)
    with col2:
        NumberOfAddress = st.number_input("Number of Addresses", 1, 10, 1)
        Complain = st.selectbox("Complained?", ["Yes", "No"])
        OrderAmountHikeFromlastYear = st.number_input("Order Amount Hike (%)", 0, 100, 20)
        CouponUsed = st.number_input("Coupons Used", 0, 100, 5)
        OrderCount = st.number_input("Order Count", 0, 100, 10)
        DaySinceLastOrder = st.number_input("Days Since Last Order", 0, 365, 20)
        CashbackAmount = st.number_input("Cashback Amount", 0, 10000, 200)

    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married"])
    PreferedOrderCat = st.selectbox("Preferred Order Category", ["Grocery", "Laptop & Accessory", "Mobile", "Mobile Phone", "Others"])
    PreferredLoginDevice = st.selectbox("Preferred Login Device", ["Mobile Phone", "Phone"])
    PreferredPaymentMode = st.selectbox("Preferred Payment Mode", ["COD (Cash on Delivery)", "Credit Card", "Debit Card", "E wallet", "UPI"])

    submit = st.form_submit_button("Predict Now 🔮")

if submit:
    input_data = pd.DataFrame({
        "Age": [Age],
        "Tenure": [Tenure],
        "CityTier": [CityTier],
        "WarehouseToHome": [WarehouseToHome],
        "HourSpendOnApp": [HourSpendOnApp],
        "NumberOfDeviceRegistered": [NumberOfDeviceRegistered],
        "SatisfactionScore": [SatisfactionScore],
        "NumberOfAddress": [NumberOfAddress],
        "Complain": [1 if Complain == "Yes" else 0],
        "OrderAmountHikeFromlastYear": [OrderAmountHikeFromlastYear],
        "CouponUsed": [CouponUsed],
        "OrderCount": [OrderCount],
        "DaySinceLastOrder": [DaySinceLastOrder],
        "CashbackAmount": [CashbackAmount],
        "Gender": [Gender],
        "MaritalStatus": [MaritalStatus],
        "PreferedOrderCat": [PreferedOrderCat],
        "PreferredLoginDevice": [PreferredLoginDevice],
        "PreferredPaymentMode": [PreferredPaymentMode]
    })

    input_processed = preprocess(input_data)

    scaled = scaler.transform(input_processed)
    prob = model.predict_proba(scaled)[0][1]
    prediction = int(prob > 0.5)

    st.subheader("📈 Churn Risk Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "red" if prob > 0.7 else "orange" if prob > 0.4 else "green"},
            'steps': [
                {'range': [0, 0.4], 'color': "lightgreen"},
                {'range': [0.4, 0.7], 'color': "orange"},
                {'range': [0.7, 1], 'color': "red"}
            ]
        },
        number={"valueformat": ".2%"},
        title={'text': "Churn Probability"}
    ))
    st.plotly_chart(fig, use_container_width=True)

    if prediction == 1:
        st.warning(f"⚠️ Customer likely to churn. | Probability: {prob:.2f}")
    else:
        st.success(f"✅ Customer not likely to churn. | Probability: {(1 - prob):.2f}")
