import streamlit as st
from supabase import create_client
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Supabase credentials
SUPABASE_URL = "https://kyanyydvgfdjsirsjmvr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt5YW55eWR2Z2ZkanNpcnNqbXZyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTAyNTg4MzgsImV4cCI6MjA2NTgzNDgzOH0.GkZXkMoZgP5hx8sz7fh4LVaXIa-MHzbwJPup_ZKSiTs"

# Supabase setup
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- AUTH LOGIC ---
def register_user(email, password):
    try:
        result = supabase.auth.sign_up({"email": email, "password": password})
        user_id = result.user.id
        supabase.table("profiles").insert({"id": user_id, "email": email, "role": "user"}).execute()
        return True, "User registered successfully."
    except Exception as e:
        return False, str(e)

def login_user(email, password):
    try:
        result = supabase.auth.sign_in_with_password({"email": email, "password": password})
        return result
    except Exception:
        return None

def get_user_role(user_id):
    try:
        response = supabase.table("profiles").select("role").eq("id", user_id).single().execute()
        return response.data.get("role", "user")
    except Exception:
        return "user"

# --- SESSION SETUP ---
if "user" not in st.session_state:
    st.session_state.user = None
if "user_role" not in st.session_state:
    st.session_state.user_role = None

# --- AUTH INTERFACE ---
if not st.session_state.user:
    st.set_page_config(page_title="🔐 Smart Energy Advisor+")
    st.title("🔐 Smart Energy Advisor+")
    choice = st.radio("Choose:", ["Login", "Register"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if choice == "Login":
        if st.button("Login"):
            result = login_user(email, password)
            if result:
                st.session_state.user = result.user
                st.session_state.user_role = get_user_role(result.user.id)
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error("Invalid credentials.")
    else:
        if st.button("Register"):
            success, msg = register_user(email, password)
            if success:
                st.success(str(msg))
            else:
                st.error(str(msg))

    st.stop()

# --- MAIN DASHBOARD ---
st.set_page_config(page_title="Smart Energy Advisor+", layout="wide")
st.title("🔋 Smart Energy Advisor+")
st.subheader(f"Welcome {st.session_state.user.email} | Role: {st.session_state.user_role}")

# Logout Button
with st.sidebar:
    st.markdown("---")
    if st.button("🚪 Logout"):
        supabase.auth.sign_out()
        st.session_state.user = None
        st.session_state.user_role = None
        st.experimental_rerun()

# Load Model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("model.pkl")

model = load_model()
shap.initjs()

# Upload Section
st.sidebar.header("📄 Upload Energy Data (Optional)")
uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])

# Preprocess
def load_and_preprocess(file_path_or_buffer):
    df = pd.read_csv(file_path_or_buffer, sep=';', na_values='?', low_memory=False)
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
    df['Global_intensity'] = pd.to_numeric(df['Global_intensity'], errors='coerce')
    df['Sub_metering_1'] = pd.to_numeric(df['Sub_metering_1'], errors='coerce')
    df['Sub_metering_2'] = pd.to_numeric(df['Sub_metering_2'], errors='coerce')
    df['Sub_metering_3'] = pd.to_numeric(df['Sub_metering_3'], errors='coerce')
    df.dropna(inplace=True)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    return df

# Predict Function
def predict_energy(hour, day_of_week, voltage, global_intensity, is_weekend, month, sm1, sm2, sm3):
    input_df = pd.DataFrame([{
        'hour': hour,
        'day_of_week': day_of_week,
        'Voltage': voltage,
        'Global_intensity': global_intensity,
        'is_weekend': is_weekend,
        'month': month,
        'Sub_metering_1': sm1,
        'Sub_metering_2': sm2,
        'Sub_metering_3': sm3
    }])
    return model.predict(input_df)[0]

default_values = {
    'Global_intensity': 19.8,
    'Sub_metering_1': 1.0,
    'Sub_metering_2': 1.0,
    'Sub_metering_3': 6.0
}

# Simulation Inputs
st.sidebar.header("⚙️ What-If Parameter Inputs")
selected_date = st.sidebar.date_input("Select Date", value=datetime.today())
hour = st.sidebar.slider("Hour", 0, 23, 12)
voltage = st.sidebar.slider("Voltage (V)", 220, 250, 230)
reduction_percent = st.sidebar.slider("Simulate Voltage Reduction (%)", 0, 20, 10)

# Derived Features
day_of_week = selected_date.weekday()
month = selected_date.month
is_weekend = 1 if day_of_week >= 5 else 0
if not (200 <= voltage <= 260):
    st.sidebar.warning("⚠️ Voltage outside expected range (200V–260V)!")

# Simulation Prediction
if st.sidebar.button("🔮 Predict Energy Usage"):
    current_pred = predict_energy(hour, day_of_week, voltage,
                                  default_values['Global_intensity'], is_weekend, month,
                                  default_values['Sub_metering_1'], default_values['Sub_metering_2'], default_values['Sub_metering_3'])

    reduced_voltage = voltage * (1 - reduction_percent / 100)
    reduced_pred = predict_energy(hour, day_of_week, reduced_voltage,
                                  default_values['Global_intensity'], is_weekend, month,
                                  default_values['Sub_metering_1'], default_values['Sub_metering_2'], default_values['Sub_metering_3'])

    cost_rate = 0.15
    current_cost = current_pred * cost_rate
    reduced_cost = reduced_pred * cost_rate
    savings_kwh = current_pred - reduced_pred
    savings_dollars = current_cost - reduced_cost

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔌 Current Usage (kWh)", f"{current_pred:.3f}")
        st.metric("💵 Cost", f"${current_cost:.2f}")
    with col2:
        st.metric("🔋 Reduced Usage (kWh)", f"{reduced_pred:.3f}")
        st.metric("⚡ Reduced Voltage", f"{reduced_voltage:.1f} V")
    st.success(f"💡 Potential Savings: {savings_kwh:.3f} kWh (${savings_dollars:.2f})")

# File or default
if uploaded_file:
    df = load_and_preprocess(uploaded_file)
else:
    df = load_and_preprocess("energy_data.txt")

# Model Evaluation
features = [
    'hour', 'day_of_week', 'Voltage', 'Global_intensity', 'is_weekend', 'month',
    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
]
X = df[features]
y_true = df['Global_active_power']
y_pred = model.predict(X)
df['Predicted_Power'] = y_pred

# Metrics
st.header("📊 Model Performance Metrics")
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
adjusted_r2 = 1 - (1 - r2) * ((len(y_true) - 1) / (len(y_true) - X.shape[1] - 1))

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("✅ MAE", f"{mae:.3f}")
col2.metric("✅ MSE", f"{mse:.3f}")
col3.metric("✅ RMSE", f"{rmse:.3f}")
col4.metric("✅ R²", f"{r2:.3f}")
col5.metric("✅ Adjusted R²", f"{adjusted_r2:.3f}")

# Evaluation Plots
st.subheader("📈 Evaluation Plots")
fig1 = plt.figure(figsize=(6, 4))
plt.bar(['MAE', 'MSE', 'RMSE', 'R²'], [mae, mse, rmse, r2])
st.pyplot(fig1)

fig2 = plt.figure(figsize=(8, 5))
plt.scatter(y_true[:500], y_pred[:500], alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
st.pyplot(fig2)

fig3 = plt.figure(figsize=(8, 5))
residuals = y_true - y_pred
plt.hist(residuals, bins=50, edgecolor='black')
st.pyplot(fig3)

# SHAP Explainability
st.header("🔍 SHAP Explainability")
if st.checkbox("Enable SHAP Summary"):
    X_sample = X.sample(n=min(10, len(X)), random_state=42)
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)

    st.subheader("📌 SHAP Beeswarm Plot")
    plt.clf()
    shap.summary_plot(shap_values, X_sample, show=False)
    st.pyplot(plt.gcf())

    st.subheader("📌 SHAP Feature Importance (Bar)")
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    st.pyplot(plt.gcf())

    st.subheader("📌 SHAP Waterfall Plot (1st Sample)")
    expected_val = explainer.expected_value
    if isinstance(expected_val, (list, np.ndarray)):
        expected_val = expected_val[0]
    shap_val = shap_values[0].values
    feature_vals = X_sample.iloc[0].values
    feature_names = list(X_sample.columns)
    plt.clf()
    shap.plots._waterfall.waterfall_legacy(
        expected_val, shap_val, feature_vals, feature_names=feature_names, max_display=10
    )
    st.pyplot(plt.gcf())

# Recommendations
st.header("💡 Smart Recommendations")
if (df['Voltage'] > 245).any():
    st.warning("⚠️ High voltage detected.")
if df['hour'].isin([17, 18, 19]).any():
    st.info("⏱️ Peak hours detected (5–8PM).")
if (df['Predicted_Power'] > df['Voltage'] * 0.02).any():
    st.success("✅ High power devices detected. Optimize usage.")

# Line Chart
st.header("📈 Predicted vs Actual Energy Usage")
plot_df = df[['timestamp', 'Global_active_power', 'Predicted_Power']].set_index('timestamp')
st.line_chart(plot_df.tail(168))

# Footer
st.markdown("---")
st.caption("🔬 Developed by Grace & Happy | CDUT AI Summer School Program | 2025")
