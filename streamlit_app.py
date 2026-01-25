import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
import os
import json

# ============================================
# FEATURE DEFINITIONS (must be early for reference)
# ============================================
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

# ============================================
# LOCATION & ENVIRONMENTAL DATA FETCHING
# ============================================

def get_user_location():
    """Get user's location from IP address"""
    try:
        # Get IP-based location
        response = requests.get('https://ipapi.co/json/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                'ip': data.get('ip', 'Unknown'),
                'city': data.get('city', 'Unknown'),
                'region': data.get('region', 'Unknown'),
                'country': data.get('country_name', 'Unknown'),
                'latitude': data.get('latitude', 46.8),
                'longitude': data.get('longitude', 9.8),
                'timezone': data.get('timezone', 'UTC')
            }
    except Exception as e:
        st.warning(f"Could not fetch location: {e}")
    
    # Default to Swiss Alps if location fetch fails
    return {
        'ip': 'Unknown',
        'city': 'Davos',
        'region': 'Graubünden',
        'country': 'Switzerland',
        'latitude': 46.8,
        'longitude': 9.8,
        'timezone': 'Europe/Zurich'
    }

def fetch_weather_data(lat, lon):
    """
    Fetch real-time weather and environmental data from Open-Meteo API
    (Free, no API key required, satellite-derived data)
    """
    try:
        # Current weather
        current_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,rain,snowfall,snow_depth,weather_code,surface_pressure,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,precipitation,rain,snowfall,snow_depth,shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,terrestrial_radiation&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,snowfall_sum,shortwave_radiation_sum&timezone=auto&past_days=3&forecast_days=1"
        
        response = requests.get(current_url, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Weather API returned status {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

def fetch_snow_data(lat, lon):
    """
    Fetch snow-specific data from Open-Meteo's snow/mountain API
    """
    try:
        # ERA5 reanalysis for more detailed snow data
        snow_url = f"https://archive-api.open-meteo.com/v1/era5?latitude={lat}&longitude={lon}&start_date={(datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')}&end_date={datetime.now().strftime('%Y-%m-%d')}&hourly=temperature_2m,snow_depth,surface_pressure&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum,shortwave_radiation_sum"
        
        response = requests.get(snow_url, timeout=10)
        
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def process_environmental_data(weather_data, snow_data=None):
    """
    Process raw API data into model input features
    """
    inputs = {}
    
    if weather_data is None:
        return None
    
    try:
        current = weather_data.get('current', {})
        hourly = weather_data.get('hourly', {})
        daily = weather_data.get('daily', {})
        
        # Current hour index
        now = datetime.now()
        hour_idx = now.hour
        
        # Get hourly arrays
        temps = hourly.get('temperature_2m', [0]*72)
        precip = hourly.get('precipitation', [0]*72)
        rain = hourly.get('rain', [0]*72)
        snow = hourly.get('snowfall', [0]*72)
        snow_depth = hourly.get('snow_depth', [0]*72)
        sw_rad = hourly.get('shortwave_radiation', [0]*72)
        direct_rad = hourly.get('direct_radiation', [0]*72)
        diffuse_rad = hourly.get('diffuse_radiation', [0]*72)
        
        # Daily arrays (past 3 days + today)
        daily_temps_max = daily.get('temperature_2m_max', [0]*4)
        daily_temps_min = daily.get('temperature_2m_min', [0]*4)
        daily_precip = daily.get('precipitation_sum', [0]*4)
        daily_snow = daily.get('snowfall_sum', [0]*4)
        daily_sw_rad = daily.get('shortwave_radiation_sum', [0]*4)
        
        # === TEMPERATURE ===
        inputs['TA'] = current.get('temperature_2m', temps[hour_idx] if len(temps) > hour_idx else 0)
        inputs['TA_daily'] = np.mean(daily_temps_max[-1:] + daily_temps_min[-1:]) / 2 if daily_temps_max else inputs['TA']
        inputs['TSS_mod'] = min(inputs['TA'] - 2, 0)  # Estimate snow surface temp (always <= 0)
        
        # === TIME ===
        inputs['profile_time'] = now.hour
        
        # === PRECIPITATION & SNOW ===
        current_snow_depth = current.get('snow_depth', 0) or 0
        inputs['max_height'] = current_snow_depth / 100  # Convert cm to m
        
        # Height changes (from snow depth history)
        if len(snow_depth) >= 72:
            inputs['max_height_1_diff'] = (snow_depth[-1] - snow_depth[-25]) / 100 if snow_depth[-25] else 0
            inputs['max_height_2_diff'] = (snow_depth[-1] - snow_depth[-49]) / 100 if snow_depth[-49] else 0
            inputs['max_height_3_diff'] = (snow_depth[-1] - snow_depth[-72]) / 100 if len(snow_depth) >= 72 else 0
        else:
            inputs['max_height_1_diff'] = 0
            inputs['max_height_2_diff'] = 0
            inputs['max_height_3_diff'] = 0
        
        # Daily SWE change (estimate from snowfall)
        inputs['SWE_daily'] = (daily_snow[-1] if daily_snow else 0) * 10  # Rough conversion
        inputs['MS_Rain_daily'] = daily.get('rain_sum', [0])[-1] if daily.get('rain_sum') else 0
        
        # === RADIATION ===
        # Current/recent radiation values
        recent_sw = sw_rad[max(0, hour_idx-6):hour_idx+1] if len(sw_rad) > hour_idx else [100]
        recent_direct = direct_rad[max(0, hour_idx-6):hour_idx+1] if len(direct_rad) > hour_idx else [50]
        recent_diffuse = diffuse_rad[max(0, hour_idx-6):hour_idx+1] if len(diffuse_rad) > hour_idx else [50]
        
        inputs['ISWR_daily'] = daily_sw_rad[-1] / 24 if daily_sw_rad and daily_sw_rad[-1] else np.mean(recent_sw)
        inputs['ISWR_h_daily'] = inputs['ISWR_daily'] * 0.9  # Approximate horizontal
        inputs['ISWR_dir_daily'] = np.mean(recent_direct) if recent_direct else inputs['ISWR_daily'] * 0.6
        inputs['ISWR_diff_daily'] = np.mean(recent_diffuse) if recent_diffuse else inputs['ISWR_daily'] * 0.4
        
        # Longwave radiation (estimated from temperature using Stefan-Boltzmann)
        temp_k = inputs['TA'] + 273.15
        sigma = 5.67e-8
        emissivity_sky = 0.7 + 0.003 * max(current.get('relative_humidity_2m', 70), 0)
        inputs['ILWR'] = emissivity_sky * sigma * (temp_k ** 4)  # Incoming LW
        inputs['ILWR_daily'] = inputs['ILWR']
        inputs['OLWR'] = 0.98 * sigma * ((inputs['TSS_mod'] + 273.15) ** 4)  # Outgoing LW from snow
        inputs['OLWR_daily'] = inputs['OLWR']
        
        # === HEAT FLUX (estimated) ===
        wind_speed = current.get('wind_speed_10m', 5)
        temp_diff = inputs['TA'] - inputs['TSS_mod']
        inputs['Qs'] = 1.5 * wind_speed * temp_diff  # Sensible heat estimate
        inputs['Ql'] = -0.5 * wind_speed * (1 - current.get('relative_humidity_2m', 70)/100) * 10  # Latent heat estimate
        inputs['Ql_daily'] = inputs['Ql']
        inputs['Qw_daily'] = inputs['ISWR_daily'] * 0.8  # Absorbed SW (assuming 0.8 absorptivity)
        
        # === LIQUID WATER CONTENT (estimated from conditions) ===
        # These are estimates based on temperature and precipitation patterns
        is_melting = inputs['TA'] > 0 or (inputs['TA'] > -2 and inputs['ISWR_daily'] > 200)
        
        if is_melting:
            inputs['water'] = max(0, inputs['TA'] * 5 + inputs['ISWR_daily'] * 0.1)
            inputs['mean_lwc'] = inputs['water'] / max(inputs['max_height'] * 10, 1)
        else:
            inputs['water'] = max(0, (inputs['TA'] + 5) * 2) if inputs['TA'] > -5 else 0
            inputs['mean_lwc'] = inputs['water'] / max(inputs['max_height'] * 10, 1)
        
        # Water changes (based on temperature trends)
        temp_trend_1d = inputs['TA'] - (temps[-25] if len(temps) >= 25 else inputs['TA'])
        temp_trend_2d = inputs['TA'] - (temps[-49] if len(temps) >= 49 else inputs['TA'])
        temp_trend_3d = inputs['TA'] - (temps[-72] if len(temps) >= 72 else inputs['TA'])
        
        inputs['water_1_diff'] = temp_trend_1d * 3 if is_melting else 0
        inputs['water_2_diff'] = temp_trend_2d * 3 if is_melting else 0
        inputs['water_3_diff'] = temp_trend_3d * 3 if is_melting else 0
        inputs['mean_lwc_2_diff'] = temp_trend_2d * 0.5
        inputs['mean_lwc_3_diff'] = temp_trend_3d * 0.5
        
        inputs['max_lwc'] = inputs['mean_lwc'] * 1.5
        inputs['std_lwc'] = inputs['mean_lwc'] * 0.3
        inputs['prop_up'] = 0.3 if is_melting else 0.1
        inputs['prop_wet_2_diff'] = 0.1 if temp_trend_2d > 2 else -0.05 if temp_trend_2d < -2 else 0
        inputs['sum_up'] = inputs['water'] * inputs['prop_up']
        
        # === WET LAYER DEPTH ===
        inputs['lowest_2_diff'] = 0.1 if is_melting and temp_trend_2d > 0 else 0
        inputs['lowest_3_diff'] = 0.15 if is_melting and temp_trend_3d > 0 else 0
        
        # === STABILITY INDEX (estimated) ===
        # Lower stability with: warming, rain, rapid snow loading
        base_stability = 2.5
        
        # Temperature effects
        if inputs['TA'] > 2:
            base_stability -= 0.8
        elif inputs['TA'] > 0:
            base_stability -= 0.4
        
        # Rain on snow
        if inputs['MS_Rain_daily'] > 5:
            base_stability -= 0.6
        elif inputs['MS_Rain_daily'] > 0:
            base_stability -= 0.3
        
        # New snow loading
        if inputs['max_height_1_diff'] > 0.3:
            base_stability -= 0.5
        elif inputs['max_height_1_diff'] > 0.15:
            base_stability -= 0.3
        
        # Wind effect
        if wind_speed > 15:
            base_stability -= 0.3
        
        inputs['S5'] = max(0.5, min(4.0, base_stability))
        inputs['S5_daily'] = -0.2 if temp_trend_1d > 3 else 0.1 if temp_trend_1d < -3 else 0
        
        return inputs
        
    except Exception as e:
        st.error(f"Error processing environmental data: {e}")
        return None

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

# ============================================
# LOCATION & AUTO-FETCH SECTION
# ============================================
st.subheader("📍 Location & Data Source")

col_loc1, col_loc2 = st.columns([2, 1])

with col_loc1:
    data_source = st.radio(
        "How would you like to input data?",
        ["🛰️ Auto-fetch from satellites (using my location)", "✍️ Manual input"],
        horizontal=True
    )

# Initialize location in session state
if 'location' not in st.session_state:
    st.session_state.location = None
if 'env_data' not in st.session_state:
    st.session_state.env_data = None
if 'weather_raw' not in st.session_state:
    st.session_state.weather_raw = None

if data_source == "🛰️ Auto-fetch from satellites (using my location)":
    
    with col_loc2:
        fetch_location = st.button("🔄 Refresh Location & Data", type="secondary")
    
    # Auto-fetch on first load or when button pressed
    if fetch_location or st.session_state.location is None:
        with st.spinner("📡 Fetching your location from IP address..."):
            st.session_state.location = get_user_location()
        
        with st.spinner("🛰️ Downloading satellite & weather data..."):
            lat = st.session_state.location['latitude']
            lon = st.session_state.location['longitude']
            st.session_state.weather_raw = fetch_weather_data(lat, lon)
            snow_data = fetch_snow_data(lat, lon)
            st.session_state.env_data = process_environmental_data(st.session_state.weather_raw, snow_data)
    
    # Display location info
    if st.session_state.location:
        loc = st.session_state.location
        
        st.success(f"""
        **📍 Detected Location:** {loc['city']}, {loc['region']}, {loc['country']}  
        **🌐 Coordinates:** {loc['latitude']:.4f}°N, {loc['longitude']:.4f}°E  
        **🖥️ IP Address:** {loc['ip']}  
        **🕐 Timezone:** {loc['timezone']}
        """)
        
        # Option to manually adjust coordinates
        with st.expander("🎯 Adjust Location Manually"):
            col_coord1, col_coord2 = st.columns(2)
            with col_coord1:
                new_lat = st.number_input("Latitude", value=loc['latitude'], min_value=-90.0, max_value=90.0, step=0.01)
            with col_coord2:
                new_lon = st.number_input("Longitude", value=loc['longitude'], min_value=-180.0, max_value=180.0, step=0.01)
            
            if st.button("📡 Fetch Data for New Coordinates"):
                with st.spinner("🛰️ Downloading data for new location..."):
                    st.session_state.location['latitude'] = new_lat
                    st.session_state.location['longitude'] = new_lon
                    st.session_state.weather_raw = fetch_weather_data(new_lat, new_lon)
                    snow_data = fetch_snow_data(new_lat, new_lon)
                    st.session_state.env_data = process_environmental_data(st.session_state.weather_raw, snow_data)
                    st.rerun()
    
    # Display fetched data summary
    if st.session_state.env_data:
        st.markdown("### 🛰️ Satellite & Weather Data Retrieved")
        
        env = st.session_state.env_data
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🌡️ Air Temp", f"{env.get('TA', 0):.1f}°C")
        with col2:
            st.metric("❄️ Snow Depth", f"{env.get('max_height', 0)*100:.0f} cm")
        with col3:
            st.metric("☀️ Solar Radiation", f"{env.get('ISWR_daily', 0):.0f} W/m²")
        with col4:
            st.metric("⚠️ Est. Stability", f"{env.get('S5', 2.5):.1f}")
        
        # Show raw weather data in expander
        with st.expander("📊 View Raw Satellite Data"):
            if st.session_state.weather_raw:
                current = st.session_state.weather_raw.get('current', {})
                st.json(current)
        
        # Update session state inputs with fetched data
        for key, value in env.items():
            if key in features_for_input:
                st.session_state.inputs[key] = value
        
        st.info("✅ **Data loaded!** Values below have been auto-populated. You can still adjust them manually if needed.")
    
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

# Initialize session state for inputs
if 'inputs' not in st.session_state:
    st.session_state.inputs = {f: 0.0 for f in features_for_input}

# Helper function to get input value (uses session state if available from auto-fetch)
def get_input_value(key, default=0.0):
    return st.session_state.inputs.get(key, default)

# Create input sections
st.subheader("📊 Enter Snowpack & Weather Data")

# Show data source indicator
if data_source == "🛰️ Auto-fetch from satellites (using my location)" and st.session_state.env_data:
    st.caption("🛰️ **Values pre-filled from satellite data** - You can adjust them below")
else:
    st.caption("✍️ **Manual entry mode** - Enter your observations below")

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
            value=float(get_input_value('TA', 0.0)), 
            min_value=-40.0, max_value=20.0, step=0.5,
            help="Current air temperature",
            key="input_TA"
        )
        st.session_state.inputs['TA_daily'] = st.number_input(
            "Daily Avg Temperature (°C)", 
            value=float(get_input_value('TA_daily', 0.0)), 
            min_value=-40.0, max_value=20.0, step=0.5,
            help="Daily average air temperature",
            key="input_TA_daily"
        )
    
    with col2:
        st.session_state.inputs['TSS_mod'] = st.number_input(
            "Snow Surface Temp (°C)", 
            value=float(get_input_value('TSS_mod', 0.0)), 
            min_value=-40.0, max_value=0.0, step=0.5,
            help="Modeled snow surface temperature",
            key="input_TSS_mod"
        )
        st.session_state.inputs['MS_Rain_daily'] = st.number_input(
            "Daily Rainfall (kg/m²)", 
            value=float(get_input_value('MS_Rain_daily', 0.0)), 
            min_value=0.0, max_value=100.0, step=1.0,
            help="Daily rainfall mass input",
            key="input_MS_Rain_daily"
        )
    
    with col3:
        st.session_state.inputs['profile_time'] = st.slider(
            "Hour of Day", 
            min_value=0, max_value=23, 
            value=int(get_input_value('profile_time', 12)),
            help="Time of day for the observation",
            key="input_profile_time"
        )

with tab2:
    st.markdown("### Liquid Water Content")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state.inputs['water'] = st.number_input(
            "Total Liquid Water (kg/m²)", 
            value=float(get_input_value('water', 0.0)), 
            min_value=0.0, max_value=500.0, step=5.0,
            key="input_water"
        )
        st.session_state.inputs['water_1_diff'] = st.number_input(
            "Water Change 1-Day", 
            value=float(get_input_value('water_1_diff', 0.0)), 
            min_value=-100.0, max_value=100.0, step=1.0,
            key="input_water_1_diff"
        )
        st.session_state.inputs['water_2_diff'] = st.number_input(
            "Water Change 2-Day", 
            value=float(get_input_value('water_2_diff', 0.0)), 
            min_value=-200.0, max_value=200.0, step=1.0,
            key="input_water_2_diff"
        )
        st.session_state.inputs['water_3_diff'] = st.number_input(
            "Water Change 3-Day", 
            value=float(get_input_value('water_3_diff', 0.0)), 
            min_value=-300.0, max_value=300.0, step=1.0,
            key="input_water_3_diff"
        )
    
    with col2:
        st.session_state.inputs['mean_lwc'] = st.number_input(
            "Mean LWC", 
            value=float(get_input_value('mean_lwc', 0.0)), 
            min_value=0.0, max_value=100.0, step=1.0,
            key="input_mean_lwc"
        )
        st.session_state.inputs['mean_lwc_2_diff'] = st.number_input(
            "Mean LWC Change 2-Day", 
            value=float(get_input_value('mean_lwc_2_diff', 0.0)), 
            min_value=-50.0, max_value=50.0, step=0.5,
            key="input_mean_lwc_2_diff"
        )
        st.session_state.inputs['mean_lwc_3_diff'] = st.number_input(
            "Mean LWC Change 3-Day", 
            value=float(get_input_value('mean_lwc_3_diff', 0.0)), 
            min_value=-50.0, max_value=50.0, step=0.5,
            key="input_mean_lwc_3_diff"
        )
        st.session_state.inputs['max_lwc'] = st.number_input(
            "Max LWC", 
            value=float(get_input_value('max_lwc', 0.0)), 
            min_value=0.0, max_value=100.0, step=1.0,
            key="input_max_lwc"
        )
    
    with col3:
        st.session_state.inputs['std_lwc'] = st.number_input(
            "LWC Std Dev", 
            value=float(get_input_value('std_lwc', 0.0)), 
            min_value=0.0, max_value=50.0, step=0.5,
            key="input_std_lwc"
        )
        st.session_state.inputs['prop_up'] = st.number_input(
            "Upper Wet Fraction (0-1)", 
            value=float(get_input_value('prop_up', 0.0)), 
            min_value=0.0, max_value=1.0, step=0.05,
            key="input_prop_up"
        )
        st.session_state.inputs['prop_wet_2_diff'] = st.number_input(
            "Wet Fraction Change 2-Day", 
            value=float(get_input_value('prop_wet_2_diff', 0.0)), 
            min_value=-1.0, max_value=1.0, step=0.05,
            key="input_prop_wet_2_diff"
        )
        st.session_state.inputs['sum_up'] = st.number_input(
            "Upper Layer Water", 
            value=float(get_input_value('sum_up', 0.0)), 
            min_value=0.0, max_value=100.0, step=1.0,
            key="input_sum_up"
        )

with tab3:
    st.markdown("### Radiation & Heat Flux")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Longwave Radiation**")
        st.session_state.inputs['ILWR'] = st.number_input(
            "Incoming LW (W/m²)", 
            value=float(get_input_value('ILWR', 250.0)), 
            min_value=100.0, max_value=400.0, step=5.0,
            key="input_ILWR"
        )
        st.session_state.inputs['ILWR_daily'] = st.number_input(
            "Daily Incoming LW", 
            value=float(get_input_value('ILWR_daily', 250.0)), 
            min_value=100.0, max_value=400.0, step=5.0,
            key="input_ILWR_daily"
        )
        st.session_state.inputs['OLWR'] = st.number_input(
            "Outgoing LW (W/m²)", 
            value=float(get_input_value('OLWR', 300.0)), 
            min_value=200.0, max_value=400.0, step=5.0,
            key="input_OLWR"
        )
        st.session_state.inputs['OLWR_daily'] = st.number_input(
            "Daily Outgoing LW", 
            value=float(get_input_value('OLWR_daily', 300.0)), 
            min_value=200.0, max_value=400.0, step=5.0,
            key="input_OLWR_daily"
        )
    
    with col2:
        st.markdown("**Shortwave Radiation**")
        st.session_state.inputs['ISWR_daily'] = st.number_input(
            "Daily SW Total (W/m²)", 
            value=float(get_input_value('ISWR_daily', 100.0)), 
            min_value=0.0, max_value=1000.0, step=10.0,
            key="input_ISWR_daily"
        )
        st.session_state.inputs['ISWR_h_daily'] = st.number_input(
            "Daily Horizontal SW", 
            value=float(get_input_value('ISWR_h_daily', 100.0)), 
            min_value=0.0, max_value=1000.0, step=10.0,
            key="input_ISWR_h_daily"
        )
        st.session_state.inputs['ISWR_dir_daily'] = st.number_input(
            "Daily Direct SW", 
            value=float(get_input_value('ISWR_dir_daily', 50.0)), 
            min_value=0.0, max_value=800.0, step=10.0,
            key="input_ISWR_dir_daily"
        )
        st.session_state.inputs['ISWR_diff_daily'] = st.number_input(
            "Daily Diffuse SW", 
            value=float(get_input_value('ISWR_diff_daily', 50.0)), 
            min_value=0.0, max_value=500.0, step=10.0,
            key="input_ISWR_diff_daily"
        )
    
    with col3:
        st.markdown("**Heat Flux**")
        st.session_state.inputs['Qs'] = st.number_input(
            "Sensible Heat (W/m²)", 
            value=float(get_input_value('Qs', 0.0)), 
            min_value=-200.0, max_value=200.0, step=5.0,
            key="input_Qs"
        )
        st.session_state.inputs['Ql'] = st.number_input(
            "Latent Heat (W/m²)", 
            value=float(get_input_value('Ql', 0.0)), 
            min_value=-200.0, max_value=200.0, step=5.0,
            key="input_Ql"
        )
        st.session_state.inputs['Ql_daily'] = st.number_input(
            "Daily Latent Heat", 
            value=float(get_input_value('Ql_daily', 0.0)), 
            min_value=-200.0, max_value=200.0, step=5.0,
            key="input_Ql_daily"
        )
        st.session_state.inputs['Qw_daily'] = st.number_input(
            "Daily Absorbed SW", 
            value=float(get_input_value('Qw_daily', 50.0)), 
            min_value=0.0, max_value=500.0, step=10.0,
            key="input_Qw_daily"
        )

with tab4:
    st.markdown("### Snow Properties")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Snow Height**")
        st.session_state.inputs['max_height'] = st.number_input(
            "Snow Height (m)", 
            value=float(get_input_value('max_height', 1.0)), 
            min_value=0.0, max_value=10.0, step=0.1,
            key="input_max_height"
        )
        st.session_state.inputs['max_height_1_diff'] = st.number_input(
            "Height Change 1-Day (m)", 
            value=float(get_input_value('max_height_1_diff', 0.0)), 
            min_value=-1.0, max_value=1.0, step=0.05,
            key="input_max_height_1_diff"
        )
        st.session_state.inputs['max_height_2_diff'] = st.number_input(
            "Height Change 2-Day (m)", 
            value=float(get_input_value('max_height_2_diff', 0.0)), 
            min_value=-2.0, max_value=2.0, step=0.05,
            key="input_max_height_2_diff"
        )
        st.session_state.inputs['max_height_3_diff'] = st.number_input(
            "Height Change 3-Day (m)", 
            value=float(get_input_value('max_height_3_diff', 0.0)), 
            min_value=-3.0, max_value=3.0, step=0.05,
            key="input_max_height_3_diff"
        )
    
    with col2:
        st.markdown("**Other Properties**")
        st.session_state.inputs['SWE_daily'] = st.number_input(
            "Daily SWE Change (mm)", 
            value=float(get_input_value('SWE_daily', 0.0)), 
            min_value=-50.0, max_value=100.0, step=1.0,
            key="input_SWE_daily"
        )
        st.session_state.inputs['lowest_2_diff'] = st.number_input(
            "Deepest Layer Change 2-Day", 
            value=float(get_input_value('lowest_2_diff', 0.0)), 
            min_value=-1.0, max_value=1.0, step=0.05,
            key="input_lowest_2_diff"
        )
        st.session_state.inputs['lowest_3_diff'] = st.number_input(
            "Deepest Layer Change 3-Day", 
            value=float(get_input_value('lowest_3_diff', 0.0)), 
            min_value=-2.0, max_value=2.0, step=0.05,
            key="input_lowest_3_diff"
        )

with tab5:
    st.markdown("### Stability Indicators")
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.inputs['S5'] = st.number_input(
            "Stability Index (S5)", 
            value=float(get_input_value('S5', 1.5)), 
            min_value=0.0, max_value=5.0, step=0.1,
            help="Skier stability index - lower values indicate less stable conditions",
            key="input_S5"
        )
    
    with col2:
        st.session_state.inputs['S5_daily'] = st.number_input(
            "Daily Stability Change", 
            value=float(get_input_value('S5_daily', 0.0)), 
            min_value=-2.0, max_value=2.0, step=0.1,
            help="Change in stability index over the day",
            key="input_S5_daily"
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

# Footer with data sources
st.markdown("---")

# Data source information (only show when auto-fetch is active)
if 'env_data' in st.session_state and st.session_state.env_data:
    with st.expander("📡 Data Sources & Attribution"):
        st.markdown("""
        ### Satellite & Weather Data Sources
        
        | Data Type | Source | Coverage |
        |-----------|--------|----------|
        | 🌡️ Temperature | [Open-Meteo API](https://open-meteo.com/) | Global, hourly |
        | ☀️ Solar Radiation | Open-Meteo (ERA5 reanalysis) | Global, daily |
        | 🌧️ Precipitation | Open-Meteo | Global, hourly |
        | ❄️ Snow Depth | Open-Meteo (ERA5) | Global, daily |
        | 📍 Geolocation | [ipapi.co](https://ipapi.co/) | IP-based |
        
        ### Data Limitations
        - **Snow-specific parameters** (liquid water content, stability indices) are **estimated** from available data
        - Weather data is satellite-derived and may differ from on-ground measurements
        - Snow depth data is from reanalysis models, not direct observation
        - For critical decisions, use official avalanche forecasts from local authorities
        
        ### Real-Time vs Estimated
        - ✅ **Direct from satellite**: Temperature, radiation, precipitation, snow depth
        - 🔶 **Physics-estimated**: Liquid water content, heat fluxes, stability indices
        - ⚠️ **Approximated**: Layer-specific properties (require snowpit observations)
        """)

st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🏔️ Avalanche Prediction System | Physics-Informed Neural Network</p>
    <p><small>⚠️ This tool provides risk estimates only. Always consult official avalanche forecasts 
    and use proper safety equipment when traveling in avalanche terrain.</small></p>
</div>
""", unsafe_allow_html=True)
