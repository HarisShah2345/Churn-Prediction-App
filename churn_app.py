
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def load_model():
    data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    data.drop('customerID', axis=1, inplace=True)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

    le = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == object:
            data[col] = le.fit_transform(data[col])

    X = data.drop('Churn', axis=1)
    y = data['Churn']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_scaled, y)
    return clf, scaler, le, X.columns.tolist()

model, scaler, le, feature_order = load_model()

st.title("📊 Customer Churn Prediction App")

with st.form("customer_form"):
    st.subheader("Enter Customer Details:")

    gender = st.selectbox("Gender", ['Male', 'Female'])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ['Yes', 'No'])
    dependents = st.selectbox("Dependents", ['Yes', 'No'])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ['Yes', 'No'])
    multiple = st.selectbox("Multiple Lines", ['Yes', 'No'])
    internet = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    online_sec = st.selectbox("Online Security", ['Yes', 'No'])
    online_bkp = st.selectbox("Online Backup", ['Yes', 'No'])
    device = st.selectbox("Device Protection", ['Yes', 'No'])
    tech = st.selectbox("Tech Support", ['Yes', 'No'])
    stream_tv = st.selectbox("Streaming TV", ['Yes', 'No'])
    stream_movies = st.selectbox("Streaming Movies", ['Yes', 'No'])
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
    payment = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total = st.number_input("Total Charges", min_value=0.0, value=1000.0)

    submit = st.form_submit_button("Predict")

if submit:
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'MultipleLines': multiple,
        'InternetService': internet,
        'OnlineSecurity': online_sec,
        'OnlineBackup': online_bkp,
        'DeviceProtection': device,
        'TechSupport': tech,
        'StreamingTV': stream_tv,
        'StreamingMovies': stream_movies,
        'Contract': contract,
        'PaperlessBilling': billing,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly,
        'TotalCharges': total
    }

    df_input = pd.DataFrame([input_data])
    for col in df_input.columns:
        if df_input[col].dtype == object:
            df_input[col] = le.fit_transform(df_input[col])
    for col in feature_order:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[feature_order]
    df_scaled = scaler.transform(df_input)
    pred = model.predict(df_scaled)[0]

    st.subheader("Prediction Result:")
    st.success("✅ This customer is **not likely** to churn." if pred == 0 else "⚠️ This customer is **likely** to churn!")
