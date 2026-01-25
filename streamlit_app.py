import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Avalanche Prediction System",
    page_icon="🏔️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #ff4444;
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
    }
    .risk-medium {
        background-color: #ffbb33;
        padding: 20px;
        border-radius: 10px;
        color: black;
        text-align: center;
        font-size: 1.5rem;
    }
    .risk-low {
        background-color: #00C851;
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏔️ Avalanche Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for model loading
st.sidebar.header("⚙️ Model Settings")
st.sidebar.info("""
This app predicts avalanche risk based on snowpack and weather conditions 
using a Physics-Informed Neural Network (PINN).
""")

# Feature descriptions
feature_info = {
    # Liquid water content
    "water": ("Total Liquid Water (kg/m²)", "Total liquid water content in the snowpack"),
    "water_1_diff": ("Water Change 1-Day", "Change in total liquid water over the last 1 day"),
    "water_2_diff": ("Water Change 2-Day", "Change in total liquid water over the last 2 days"),
    "water_3_diff": ("Water Change 3-Day", "Change in total liquid water over the last 3 days"),
    "mean_lwc": ("Mean LWC", "Average liquid water content across layers"),
    "mean_lwc_2_diff": ("Mean LWC Change 2-Day", "Change in mean liquid water over 2 days"),
    "mean_lwc_3_diff": ("Mean LWC Change 3-Day", "Change in mean liquid water over 3 days"),
    "max_lwc": ("Max LWC", "Maximum liquid water present in any layer"),
    "std_lwc": ("LWC Std Dev", "Standard deviation of liquid water between layers"),
    
    # Wetness distribution
    "prop_up": ("Upper Wet Fraction", "Wet fraction in top 15 cm"),
    "prop_wet_2_diff": ("Wet Fraction Change 2-Day", "Change in wet fraction over 2 days"),
    "sum_up": ("Upper Layer Water", "Total liquid water in top 15 cm"),
    
    # Wet layer depth
    "lowest_2_diff": ("Deepest Layer Change 2-Day", "Change in deepest wet layer depth over 2 days"),
    "lowest_3_diff": ("Deepest Layer Change 3-Day", "Change in deepest wet layer depth over 3 days"),
    
    # Temperature
    "TA": ("Air Temperature (°C)", "Current air temperature"),
    "TA_daily": ("Daily Avg Temperature (°C)", "Daily average air temperature"),
    "TSS_mod": ("Snow Surface Temp (°C)", "Modeled snow surface temperature"),
    
    # Radiation
    "ILWR": ("Incoming LW Radiation", "Incoming longwave radiation (W/m²)"),
    "ILWR_daily": ("Daily Incoming LW", "Daily total incoming longwave radiation"),
    "ISWR_daily": ("Daily SW Radiation", "Daily total shortwave radiation"),
    "ISWR_h_daily": ("Daily Horizontal SW", "Daily horizontal shortwave radiation"),
    "ISWR_dir_daily": ("Daily Direct SW", "Daily direct shortwave radiation"),
    "ISWR_diff_daily": ("Daily Diffuse SW", "Daily diffuse shortwave radiation"),
    "OLWR": ("Outgoing LW Radiation", "Outgoing longwave radiation (W/m²)"),
    "OLWR_daily": ("Daily Outgoing LW", "Daily outgoing longwave radiation"),
    
    # Heat flux
    "Qs": ("Sensible Heat Flux", "Sensible heat flux (W/m²)"),
    "Ql": ("Latent Heat Flux", "Latent heat flux related to melt and refreeze"),
    "Ql_daily": ("Daily Latent Heat", "Daily latent heat flux"),
    "Qw_daily": ("Daily Absorbed SW", "Daily absorbed shortwave energy"),
    
    # Snow properties
    "SWE_daily": ("Daily SWE Change", "Daily change in snow water equivalent"),
    "MS_Rain_daily": ("Daily Rainfall", "Daily rainfall mass input (kg/m²)"),
    "max_height": ("Snow Height (m)", "Maximum modeled snow height"),
    "max_height_1_diff": ("Height Change 1-Day", "Change in snow height over 1 day"),
    "max_height_2_diff": ("Height Change 2-Day", "Change in snow height over 2 days"),
    "max_height_3_diff": ("Height Change 3-Day", "Change in snow height over 3 days"),
    
    # Stability
    "S5": ("Stability Index", "Skier stability index (lower = less stable)"),
    "S5_daily": ("Stability Change", "Daily change in skier stability index"),
    
    # Time
    "profile_time": ("Hour of Day", "Time of day (0-23)"),
}

# Ordered features list (must match model training order)
features_for_input = [
    'mean_lwc_3_diff', 'OLWR_daily', 'max_height_1_diff', 'sum_up',
    'ISWR_h_daily', 'max_lwc', 'S5_daily', 'mean_lwc', 'water_2_diff',
    'TA_daily', 'Ql', 'ISWR_daily', 'SWE_daily', 'ILWR_daily', 'profile_time',
    'Qw_daily', 'OLWR', 'ILWR', 'Ql_daily', 'prop_up', 'ISWR_dir_daily',
    'water_1_diff', 'TSS_mod', 'lowest_3_diff', 'max_height_2_diff', 'max_height',
    'water', 'prop_wet_2_diff', 'MS_Rain_daily', 'water_3_diff', 'std_lwc',
    'mean_lwc_2_diff', 'Qs', 'max_height_3_diff', 'S5', 'TA', 'lowest_2_diff',
    'ISWR_diff_daily'
]

# Initialize session state for inputs
if 'inputs' not in st.session_state:
    st.session_state.inputs = {f: 0.0 for f in features_for_input}

# Create input sections
st.subheader("📊 Enter Snowpack & Weather Data")

# Use tabs for organized input
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌡️ Temperature & Weather", 
    "💧 Liquid Water Content",
    "☀️ Radiation",
    "📏 Snow Properties",
    "⚠️ Stability"
])

with tab1:
    st.markdown("### Temperature & Weather Conditions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state.inputs['TA'] = st.number_input(
            "Air Temperature (°C)", 
            value=0.0, min_value=-40.0, max_value=20.0, step=0.5,
            help="Current air temperature"
        )
        st.session_state.inputs['TA_daily'] = st.number_input(
            "Daily Avg Temperature (°C)", 
            value=0.0, min_value=-40.0, max_value=20.0, step=0.5,
            help="Daily average air temperature"
        )
    
    with col2:
        st.session_state.inputs['TSS_mod'] = st.number_input(
            "Snow Surface Temp (°C)", 
            value=0.0, min_value=-40.0, max_value=0.0, step=0.5,
            help="Modeled snow surface temperature"
        )
        st.session_state.inputs['MS_Rain_daily'] = st.number_input(
            "Daily Rainfall (kg/m²)", 
            value=0.0, min_value=0.0, max_value=100.0, step=1.0,
            help="Daily rainfall mass input"
        )
    
    with col3:
        st.session_state.inputs['profile_time'] = st.slider(
            "Hour of Day", 
            min_value=0, max_value=23, value=12,
            help="Time of day for the observation"
        )

with tab2:
    st.markdown("### Liquid Water Content")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state.inputs['water'] = st.number_input(
            "Total Liquid Water (kg/m²)", 
            value=0.0, min_value=0.0, max_value=500.0, step=5.0
        )
        st.session_state.inputs['water_1_diff'] = st.number_input(
            "Water Change 1-Day", 
            value=0.0, min_value=-100.0, max_value=100.0, step=1.0
        )
        st.session_state.inputs['water_2_diff'] = st.number_input(
            "Water Change 2-Day", 
            value=0.0, min_value=-200.0, max_value=200.0, step=1.0
        )
        st.session_state.inputs['water_3_diff'] = st.number_input(
            "Water Change 3-Day", 
            value=0.0, min_value=-300.0, max_value=300.0, step=1.0
        )
    
    with col2:
        st.session_state.inputs['mean_lwc'] = st.number_input(
            "Mean LWC", 
            value=0.0, min_value=0.0, max_value=100.0, step=1.0
        )
        st.session_state.inputs['mean_lwc_2_diff'] = st.number_input(
            "Mean LWC Change 2-Day", 
            value=0.0, min_value=-50.0, max_value=50.0, step=0.5
        )
        st.session_state.inputs['mean_lwc_3_diff'] = st.number_input(
            "Mean LWC Change 3-Day", 
            value=0.0, min_value=-50.0, max_value=50.0, step=0.5
        )
        st.session_state.inputs['max_lwc'] = st.number_input(
            "Max LWC", 
            value=0.0, min_value=0.0, max_value=100.0, step=1.0
        )
    
    with col3:
        st.session_state.inputs['std_lwc'] = st.number_input(
            "LWC Std Dev", 
            value=0.0, min_value=0.0, max_value=50.0, step=0.5
        )
        st.session_state.inputs['prop_up'] = st.number_input(
            "Upper Wet Fraction (0-1)", 
            value=0.0, min_value=0.0, max_value=1.0, step=0.05
        )
        st.session_state.inputs['prop_wet_2_diff'] = st.number_input(
            "Wet Fraction Change 2-Day", 
            value=0.0, min_value=-1.0, max_value=1.0, step=0.05
        )
        st.session_state.inputs['sum_up'] = st.number_input(
            "Upper Layer Water", 
            value=0.0, min_value=0.0, max_value=100.0, step=1.0
        )

with tab3:
    st.markdown("### Radiation & Heat Flux")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Longwave Radiation**")
        st.session_state.inputs['ILWR'] = st.number_input(
            "Incoming LW (W/m²)", 
            value=250.0, min_value=100.0, max_value=400.0, step=5.0
        )
        st.session_state.inputs['ILWR_daily'] = st.number_input(
            "Daily Incoming LW", 
            value=250.0, min_value=100.0, max_value=400.0, step=5.0
        )
        st.session_state.inputs['OLWR'] = st.number_input(
            "Outgoing LW (W/m²)", 
            value=300.0, min_value=200.0, max_value=400.0, step=5.0
        )
        st.session_state.inputs['OLWR_daily'] = st.number_input(
            "Daily Outgoing LW", 
            value=300.0, min_value=200.0, max_value=400.0, step=5.0
        )
    
    with col2:
        st.markdown("**Shortwave Radiation**")
        st.session_state.inputs['ISWR_daily'] = st.number_input(
            "Daily SW Total (W/m²)", 
            value=100.0, min_value=0.0, max_value=1000.0, step=10.0
        )
        st.session_state.inputs['ISWR_h_daily'] = st.number_input(
            "Daily Horizontal SW", 
            value=100.0, min_value=0.0, max_value=1000.0, step=10.0
        )
        st.session_state.inputs['ISWR_dir_daily'] = st.number_input(
            "Daily Direct SW", 
            value=50.0, min_value=0.0, max_value=800.0, step=10.0
        )
        st.session_state.inputs['ISWR_diff_daily'] = st.number_input(
            "Daily Diffuse SW", 
            value=50.0, min_value=0.0, max_value=500.0, step=10.0
        )
    
    with col3:
        st.markdown("**Heat Flux**")
        st.session_state.inputs['Qs'] = st.number_input(
            "Sensible Heat (W/m²)", 
            value=0.0, min_value=-200.0, max_value=200.0, step=5.0
        )
        st.session_state.inputs['Ql'] = st.number_input(
            "Latent Heat (W/m²)", 
            value=0.0, min_value=-200.0, max_value=200.0, step=5.0
        )
        st.session_state.inputs['Ql_daily'] = st.number_input(
            "Daily Latent Heat", 
            value=0.0, min_value=-200.0, max_value=200.0, step=5.0
        )
        st.session_state.inputs['Qw_daily'] = st.number_input(
            "Daily Absorbed SW", 
            value=50.0, min_value=0.0, max_value=500.0, step=10.0
        )

with tab4:
    st.markdown("### Snow Properties")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Snow Height**")
        st.session_state.inputs['max_height'] = st.number_input(
            "Snow Height (m)", 
            value=1.0, min_value=0.0, max_value=10.0, step=0.1
        )
        st.session_state.inputs['max_height_1_diff'] = st.number_input(
            "Height Change 1-Day (m)", 
            value=0.0, min_value=-1.0, max_value=1.0, step=0.05
        )
        st.session_state.inputs['max_height_2_diff'] = st.number_input(
            "Height Change 2-Day (m)", 
            value=0.0, min_value=-2.0, max_value=2.0, step=0.05
        )
        st.session_state.inputs['max_height_3_diff'] = st.number_input(
            "Height Change 3-Day (m)", 
            value=0.0, min_value=-3.0, max_value=3.0, step=0.05
        )
    
    with col2:
        st.markdown("**Other Properties**")
        st.session_state.inputs['SWE_daily'] = st.number_input(
            "Daily SWE Change (mm)", 
            value=0.0, min_value=-50.0, max_value=100.0, step=1.0
        )
        st.session_state.inputs['lowest_2_diff'] = st.number_input(
            "Deepest Layer Change 2-Day", 
            value=0.0, min_value=-1.0, max_value=1.0, step=0.05
        )
        st.session_state.inputs['lowest_3_diff'] = st.number_input(
            "Deepest Layer Change 3-Day", 
            value=0.0, min_value=-2.0, max_value=2.0, step=0.05
        )

with tab5:
    st.markdown("### Stability Indicators")
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.inputs['S5'] = st.number_input(
            "Stability Index (S5)", 
            value=1.5, min_value=0.0, max_value=5.0, step=0.1,
            help="Skier stability index - lower values indicate less stable conditions"
        )
    
    with col2:
        st.session_state.inputs['S5_daily'] = st.number_input(
            "Daily Stability Change", 
            value=0.0, min_value=-2.0, max_value=2.0, step=0.1,
            help="Change in stability index over the day"
        )
    
    # Visual stability indicator
    s5_value = st.session_state.inputs['S5']
    if s5_value < 1.0:
        st.error("⚠️ Very Low Stability - High Danger!")
    elif s5_value < 1.5:
        st.warning("⚡ Low Stability - Considerable Danger")
    elif s5_value < 2.5:
        st.info("📊 Moderate Stability")
    else:
        st.success("✅ Good Stability")

st.markdown("---")

# Prediction section
st.subheader("🎯 Avalanche Risk Prediction")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button("🔮 Predict Avalanche Risk", type="primary", use_container_width=True)

if predict_button:
    # Create input dataframe in correct order
    input_data = pd.DataFrame([[st.session_state.inputs[f] for f in features_for_input]], 
                              columns=features_for_input)
    
    # Check if model files exist
    model_path = "avalanche_model"
    scaler_path = "scaler.joblib"
    imputer_path = "imputer.joblib"
    
    # For demo purposes, we'll simulate a prediction
    # In production, you would load your actual model:
    # model = tf.keras.models.load_model(model_path)
    # scaler = joblib.load(scaler_path)
    # imputer = joblib.load(imputer_path)
    
    # Simulated prediction based on key risk factors
    # Replace this with actual model prediction
    try:
        # Try to load actual model
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        imputer = joblib.load(imputer_path)
        
        # Preprocess
        input_imputed = imputer.transform(input_data)
        input_scaled = scaler.transform(input_imputed)
        
        # Predict
        prediction, _ = model.predict(input_scaled)
        confidence = float(prediction[0][0])
        
    except Exception as e:
        # Fallback: Simple heuristic-based prediction for demo
        st.warning("⚠️ Model files not found. Using simplified risk assessment.")
        
        # Calculate risk based on key factors
        risk_score = 0.3  # Base risk
        
        # Temperature factors
        if st.session_state.inputs['TA'] > 0:
            risk_score += 0.15
        if st.session_state.inputs['TA_daily'] > st.session_state.inputs['TA']:
            risk_score += 0.1  # Warming trend
        
        # Water content factors
        if st.session_state.inputs['water_1_diff'] > 10:
            risk_score += 0.15
        if st.session_state.inputs['mean_lwc'] > 20:
            risk_score += 0.1
        
        # Stability factors
        if st.session_state.inputs['S5'] < 1.0:
            risk_score += 0.25
        elif st.session_state.inputs['S5'] < 1.5:
            risk_score += 0.15
        elif st.session_state.inputs['S5'] < 2.0:
            risk_score += 0.05
        
        # Recent snowfall
        if st.session_state.inputs['max_height_1_diff'] > 0.3:
            risk_score += 0.15
        
        # Rain on snow
        if st.session_state.inputs['MS_Rain_daily'] > 5:
            risk_score += 0.2
        
        confidence = min(max(risk_score, 0.0), 1.0)
    
    # Display results
    st.markdown("---")
    
    # Main confidence display
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Determine risk level
        if confidence >= 0.7:
            risk_level = "HIGH"
            risk_class = "risk-high"
            risk_emoji = "🔴"
            risk_message = "DANGER - Avalanche conditions are likely!"
        elif confidence >= 0.4:
            risk_level = "MODERATE"
            risk_class = "risk-medium"
            risk_emoji = "🟡"
            risk_message = "CAUTION - Avalanche conditions are possible"
        else:
            risk_level = "LOW"
            risk_class = "risk-low"
            risk_emoji = "🟢"
            risk_message = "Lower risk - but always exercise caution"
        
        st.markdown(f"""
        <div class="{risk_class}">
            <h2>{risk_emoji} {risk_level} RISK</h2>
            <h3>Confidence: {confidence*100:.1f}%</h3>
            <p>{risk_message}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Progress bar visualization
    st.markdown("### Risk Level Gauge")
    st.progress(confidence)
    
    # Detailed metrics
    st.markdown("### 📈 Key Risk Factors")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Stability Index", f"{st.session_state.inputs['S5']:.2f}", 
                  delta=f"{st.session_state.inputs['S5_daily']:.2f}")
    
    with col2:
        st.metric("Air Temp", f"{st.session_state.inputs['TA']:.1f}°C",
                  delta=f"{st.session_state.inputs['TA_daily'] - st.session_state.inputs['TA']:.1f}°C")
    
    with col3:
        st.metric("Snow Height", f"{st.session_state.inputs['max_height']:.2f}m",
                  delta=f"{st.session_state.inputs['max_height_1_diff']:.2f}m")
    
    with col4:
        st.metric("Water Change", f"{st.session_state.inputs['water_1_diff']:.1f}",
                  delta="1-day change")
    
    # Safety recommendations
    st.markdown("### 🛡️ Safety Recommendations")
    
    if confidence >= 0.7:
        st.error("""
        **HIGH RISK ACTIONS:**
        - ❌ Avoid all avalanche terrain
        - 🚫 Do not travel on or below steep slopes
        - 📢 Check local avalanche advisories
        - 🏠 Consider postponing backcountry travel
        """)
    elif confidence >= 0.4:
        st.warning("""
        **MODERATE RISK ACTIONS:**
        - ⚠️ Use caution in avalanche terrain
        - 🎒 Carry avalanche safety equipment
        - 👥 Travel with partners
        - 📍 Identify safe zones and escape routes
        """)
    else:
        st.success("""
        **LOWER RISK ACTIONS:**
        - ✅ Conditions appear more stable
        - 🎒 Still carry avalanche safety gear
        - 👀 Remain vigilant for changing conditions
        - 📻 Check for updated forecasts
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🏔️ Avalanche Prediction System | Physics-Informed Neural Network</p>
    <p><small>⚠️ This tool provides risk estimates only. Always consult official avalanche forecasts 
    and use proper safety equipment when traveling in avalanche terrain.</small></p>
</div>
""", unsafe_allow_html=True)
