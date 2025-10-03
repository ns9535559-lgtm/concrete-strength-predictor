import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import xlsxwriter


# Load models and scaler
catboost_model = joblib.load('catboost_model.pkl')
xgboost_model = joblib.load('xgboost_model.pkl')
lightboost_model = joblib.load('lightgbm_model.pkl')
scaler = joblib.load('scaler.pkl')

# App title and styling
st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🧱")
st.markdown("<h1 style='text-align: left; color: #4CAF50;'>Concrete Strength Predictor</h1>", unsafe_allow_html=True)
st.markdown("""This app predicts the **compressive strength** of concrete based on its mix composition. Adjust the values below to simulate different mix designs.""")
st.markdown("---")

# Sidebar model selector
model_dict = {
    "CatBoost Model": catboost_model,
    "XGBoost Model": xgboost_model
}
selected_model_name = st.sidebar.selectbox("Choose a Model", list(model_dict.keys()))
selected_model = model_dict[selected_model_name]

# --- Input Section ---
st.subheader("Input Concrete Mix Details")

cement = st.number_input("Cement (kg/m³)", 102.0, 540.0, 320.0, step=10.0)
slag = st.number_input("Blast Furnace Slag (kg/m³)", 0.0, 360.0, 0.0, step=10.0)
fly_ash = st.number_input("Fly Ash (kg/m³)", 0.0, 200.0, 0.0, step=10.0)
water = st.number_input("Water (kg/m³)", 120.0, 250.0, 160.0, step=5.0)
superplasticizer = st.number_input("Superplasticizer (kg/m³)", 0.0, 32.0, 0.0, step=1.0)
coarse_agg = st.number_input("Coarse Aggregate (kg/m³)", 800.0, 1150.0, 1000.0, step=10.0)
fine_agg = st.number_input("Fine Aggregate (kg/m³)", 550.0, 1000.0, 800.0, step=10.0)
age = st.number_input("Age (days)", 1, 365, 28, step=1)

# Calculate Water–Cement Ratio (just for display, not passed to model)
w_c_ratio = (water / cement) if cement > 0 else 0.0
st.markdown(f"**Water–Cement Ratio:** {w_c_ratio:.2f}")

# Collect input into dictionary
user_input = {
    'Cement': cement,
    'Blast Furnace Slag': slag,
    'Fly Ash': fly_ash,
    'Water': water,
    'Superplasticizer': superplasticizer,
    'Coarse Aggregate': coarse_agg,
    'Fine Aggregate': fine_agg,
    'Age': age
}

# Correct column order & names (must match training)
feature_names = ['cement', 'slag', 'flyash', 'water',
                 'superplasticizer', 'coarseaggregate',
                 'fineaggregate', 'age']

# --- Prediction ---
if st.button("Predict Compressive Strength"):
    # Rebuild input DataFrame with current values
    input_df = pd.DataFrame([[
        cement, slag, fly_ash, water,
        superplasticizer, coarse_agg, fine_agg, age
    ]], columns=feature_names)

    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = selected_model.predict(input_scaled)[0]
    st.success(f"Predicted Compressive Strength: **{prediction:.2f} MPa**")

    # --- Stress-Strain Curve ---
    st.subheader("Stress–Strain Curve")

    def plot_stress_strain(fc, eps0=0.002, epsu=0.0035, model="hognestad"):
        strain = np.linspace(0, epsu, 200)

        if model == "hognestad":
            stress = fc * (2*(strain/eps0) - (strain/eps0)**2)
            stress[strain > eps0*2] = 0
        elif model == "linear":
            stress = np.where(strain <= eps0, fc*(strain/eps0),
                            np.where(strain <= epsu, fc*(1 - (strain-eps0)/(epsu-eps0)), 0))
        else:
            raise ValueError("Unknown model type. Choose 'hognestad' or 'linear'.")

        # Create a new figure for Streamlit each time
        fig, ax = plt.subplots(figsize=(6,5))
        ax.plot(strain, stress, linewidth=2, color="green")
        ax.set_xlabel("Strain")
        ax.set_ylabel("Stress (MPa)")
        ax.set_title(f"Concrete Stress-Strain Curve ({model.capitalize()} Model)")
        ax.grid(True)

        st.pyplot(fig)
        return strain, stress

    # Generate curve with updated prediction
    plot_stress_strain(prediction, model="hognestad")

    # --- Export Section ---
    st.subheader("Export Results")

    export_df = input_df.copy()
    export_df['Predicted Strength (MPa)'] = prediction

    # Export to Excel
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        export_df.to_excel(writer, index=False, sheet_name='Prediction')
    excel_data = excel_buffer.getvalue()
    st.download_button(
    label="Download",
    data=excel_data,
    file_name="Concrete_Prediction_Results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>© 2025 Ranti-Owoeye Victor | Powered by Machine Learning & Streamlit 🚀</div>", unsafe_allow_html=True)



