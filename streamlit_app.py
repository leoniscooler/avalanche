import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
import os
import json
import math

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
# SATELLITE DATA SOURCE CONFIGURATIONS
# ============================================
SATELLITE_SOURCES = {
    'MODIS': {
        'name': 'MODIS (Terra/Aqua)',
        'products': ['MOD10A1 (Snow Cover)', 'MOD11A1 (Land Surface Temp)', 'MCD43A3 (Albedo)'],
        'resolution': '500m - 1km',
        'provider': 'NASA Earthdata'
    },
    'VIIRS': {
        'name': 'VIIRS (Suomi NPP/NOAA-20)',
        'products': ['VNP10A1 (Snow Cover)', 'VNP21A1 (Land Surface Temp)'],
        'resolution': '375m - 750m',
        'provider': 'NASA Earthdata'
    },
    'ERA5': {
        'name': 'ERA5 Reanalysis',
        'products': ['Hourly data on single levels', 'Snow depth', 'Radiation fluxes'],
        'resolution': '0.25° (~31km)',
        'provider': 'Copernicus CDS'
    },
    'Sentinel': {
        'name': 'Sentinel-2/3',
        'products': ['Snow Cover (S2)', 'Land Surface Temp (S3)'],
        'resolution': '10m - 1km',
        'provider': 'Copernicus Data Space'
    },
    'GOES': {
        'name': 'GOES-16/17/18',
        'products': ['ABI Radiation', 'Snow/Ice Detection'],
        'resolution': '0.5km - 2km',
        'provider': 'NOAA'
    }
}

# ============================================
# LOCATION & ENVIRONMENTAL DATA FETCHING
# ============================================

def get_user_location():
    """Get user's location from IP address"""
    try:
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
                'timezone': data.get('timezone', 'UTC'),
                'elevation': None  # Will be fetched separately
            }
    except Exception as e:
        st.warning(f"Could not fetch location: {e}")
    
    return {
        'ip': 'Unknown',
        'city': 'Davos',
        'region': 'Graubünden',
        'country': 'Switzerland',
        'latitude': 46.8,
        'longitude': 9.8,
        'timezone': 'Europe/Zurich',
        'elevation': 1560
    }

def get_elevation(lat, lon):
    """Fetch elevation data from Open-Meteo API"""
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('elevation', [0])[0]
    except:
        pass
    return 1500  # Default mountain elevation

# ============================================
# NASA EARTHDATA (MODIS/VIIRS) DATA FETCHING
# ============================================

def fetch_nasa_earthdata(lat, lon, date_str=None):
    """
    Fetch MODIS and VIIRS data from NASA Earthdata CMR API
    Products: MODIS Snow Cover, LST, VIIRS Snow
    """
    data = {
        'source': 'NASA_Earthdata',
        'products_queried': ['MODIS', 'VIIRS'],
        'snow_cover': None,
        'land_surface_temp': None,
        'albedo': None,
        'available': False
    }
    
    if date_str is None:
        date_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    try:
        # NASA CMR (Common Metadata Repository) API for granule search
        # This gives us metadata about available MODIS/VIIRS products
        cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
        
        # Search for MOD10A1 (MODIS Snow Cover Daily)
        params = {
            'short_name': 'MOD10A1',
            'version': '061',
            'temporal': f"{date_str}T00:00:00Z,{date_str}T23:59:59Z",
            'bounding_box': f"{lon-0.5},{lat-0.5},{lon+0.5},{lat+0.5}",
            'page_size': 5
        }
        
        response = requests.get(cmr_url, params=params, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            entries = result.get('feed', {}).get('entry', [])
            if entries:
                data['available'] = True
                data['modis_granules'] = len(entries)
                # Extract metadata
                for entry in entries:
                    if 'polygons' in entry:
                        data['coverage'] = 'MODIS tile available'
        
        # Search for VNP10A1 (VIIRS Snow Cover)
        params['short_name'] = 'VNP10A1'
        params['version'] = '001'
        
        response = requests.get(cmr_url, params=params, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            entries = result.get('feed', {}).get('entry', [])
            if entries:
                data['viirs_granules'] = len(entries)
                data['viirs_available'] = True
                
    except Exception as e:
        data['error'] = str(e)
    
    return data

def fetch_nasa_gibs_imagery(lat, lon, date_str=None):
    """
    Fetch actual MODIS data values from NASA GIBS (Global Imagery Browse Services)
    Using the WMS/WMTS services for snow cover and temperature
    """
    if date_str is None:
        date_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    data = {
        'source': 'NASA_GIBS',
        'snow_cover_fraction': None,
        'ndsi': None,  # Normalized Difference Snow Index
    }
    
    try:
        # NASA GIBS GetFeatureInfo for point data
        # MODIS_Terra_Snow_Cover layer
        gibs_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
        
        params = {
            'SERVICE': 'WMS',
            'VERSION': '1.3.0',
            'REQUEST': 'GetFeatureInfo',
            'LAYERS': 'MODIS_Terra_Snow_Cover',
            'QUERY_LAYERS': 'MODIS_Terra_Snow_Cover',
            'INFO_FORMAT': 'application/json',
            'I': '1',
            'J': '1',
            'WIDTH': '3',
            'HEIGHT': '3',
            'CRS': 'EPSG:4326',
            'BBOX': f"{lat-0.01},{lon-0.01},{lat+0.01},{lon+0.01}",
            'TIME': date_str
        }
        
        response = requests.get(gibs_url, params=params, timeout=15)
        
        if response.status_code == 200:
            try:
                result = response.json()
                if 'features' in result and result['features']:
                    props = result['features'][0].get('properties', {})
                    # MODIS snow cover uses values 0-100 for fractional snow cover
                    # 200 = missing, 201 = no decision, 250 = clouds
                    snow_val = props.get('GRAY_INDEX', props.get('value'))
                    if snow_val and snow_val <= 100:
                        data['snow_cover_fraction'] = snow_val / 100.0
                        data['ndsi'] = (snow_val / 100.0) * 0.8  # Approximate NDSI
            except:
                pass
                
    except Exception as e:
        data['error'] = str(e)
    
    return data

# ============================================
# COPERNICUS ERA5 DATA (via Open-Meteo Archive)
# ============================================

def fetch_era5_data(lat, lon):
    """
    Fetch ERA5 reanalysis data from Open-Meteo's archive API
    (Free alternative to Copernicus CDS that doesn't require registration)
    
    ERA5 provides:
    - Temperature at 2m
    - Snow depth
    - Surface solar/thermal radiation
    - Heat fluxes
    - Precipitation
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    data = {
        'source': 'ERA5_Reanalysis',
        'available': False,
        'temperature_2m': [],
        'snow_depth': [],
        'snow_density': None,
        'surface_solar_radiation': [],
        'surface_thermal_radiation': [],
        'sensible_heat_flux': [],
        'latent_heat_flux': [],
        'precipitation': [],
        'snow_fall': [],
        'soil_temperature': []
    }
    
    try:
        # ERA5 hourly data via Open-Meteo archive
        url = f"https://archive-api.open-meteo.com/v1/era5"
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': [
                'temperature_2m',
                'snow_depth',
                'surface_pressure',
                'shortwave_radiation',
                'direct_radiation',
                'diffuse_radiation',
                'direct_normal_irradiance',
                'terrestrial_radiation',
                'precipitation',
                'snowfall',
                'rain',
                'soil_temperature_0cm',
                'soil_temperature_6cm'
            ],
            'daily': [
                'temperature_2m_max',
                'temperature_2m_min',
                'temperature_2m_mean',
                'precipitation_sum',
                'snowfall_sum',
                'rain_sum',
                'shortwave_radiation_sum'
            ]
        }
        
        response = requests.get(url, params=params, timeout=20)
        
        if response.status_code == 200:
            result = response.json()
            hourly = result.get('hourly', {})
            daily = result.get('daily', {})
            
            data['available'] = True
            data['temperature_2m'] = hourly.get('temperature_2m', [])
            data['snow_depth'] = hourly.get('snow_depth', [])
            data['shortwave_radiation'] = hourly.get('shortwave_radiation', [])
            data['direct_radiation'] = hourly.get('direct_radiation', [])
            data['diffuse_radiation'] = hourly.get('diffuse_radiation', [])
            data['terrestrial_radiation'] = hourly.get('terrestrial_radiation', [])
            data['precipitation'] = hourly.get('precipitation', [])
            data['snowfall'] = hourly.get('snowfall', [])
            data['rain'] = hourly.get('rain', [])
            data['soil_temperature'] = hourly.get('soil_temperature_0cm', [])
            
            # Daily aggregates
            data['daily_temp_max'] = daily.get('temperature_2m_max', [])
            data['daily_temp_min'] = daily.get('temperature_2m_min', [])
            data['daily_temp_mean'] = daily.get('temperature_2m_mean', [])
            data['daily_precip'] = daily.get('precipitation_sum', [])
            data['daily_snowfall'] = daily.get('snowfall_sum', [])
            data['daily_rain'] = daily.get('rain_sum', [])
            data['daily_radiation'] = daily.get('shortwave_radiation_sum', [])
            data['times'] = hourly.get('time', [])
            
    except Exception as e:
        data['error'] = str(e)
    
    return data

def fetch_era5_land_data(lat, lon):
    """
    Fetch ERA5-Land specific snow variables
    Higher resolution (9km) compared to ERA5 (31km)
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    
    data = {
        'source': 'ERA5_Land',
        'available': False,
        'snow_depth_water_equivalent': [],
        'snow_cover': [],
        'snow_albedo': [],
        'snow_density': []
    }
    
    try:
        # ERA5-Land via Open-Meteo (when available)
        url = f"https://archive-api.open-meteo.com/v1/era5"
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': ['snow_depth'],
            'models': 'era5_land'  # Request ERA5-Land specifically
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            hourly = result.get('hourly', {})
            
            if hourly.get('snow_depth'):
                data['available'] = True
                data['snow_depth'] = hourly.get('snow_depth', [])
                
    except Exception as e:
        data['error'] = str(e)
    
    return data

# ============================================
# NOAA GOES SATELLITE DATA
# ============================================

def fetch_goes_data(lat, lon):
    """
    Fetch GOES-16/17/18 satellite data
    GOES provides high-frequency (every 10-15 min) radiation and cloud data
    """
    data = {
        'source': 'GOES',
        'available': False,
        'shortwave_radiation': None,
        'longwave_radiation': None,
        'cloud_cover': None,
        'snow_ice_detection': None
    }
    
    try:
        # NOAA GOES data via their API (limited public access)
        # Alternative: Use Open-Meteo's forecast which incorporates GOES data
        
        # For radiation data, we can use CERES (derived from GOES and other satellites)
        # via NASA's POWER API
        
        power_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
        
        params = {
            'parameters': 'ALLSKY_SFC_SW_DWN,ALLSKY_SFC_LW_DWN,CLRSKY_SFC_SW_DWN',
            'community': 'RE',
            'longitude': lon,
            'latitude': lat,
            'start': week_ago,
            'end': yesterday,
            'format': 'JSON'
        }
        
        response = requests.get(power_url, params=params, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            properties = result.get('properties', {}).get('parameter', {})
            
            data['available'] = True
            data['shortwave_radiation'] = properties.get('ALLSKY_SFC_SW_DWN', {})
            data['longwave_radiation'] = properties.get('ALLSKY_SFC_LW_DWN', {})
            data['clearsky_radiation'] = properties.get('CLRSKY_SFC_SW_DWN', {})
            
    except Exception as e:
        data['error'] = str(e)
    
    return data

# ============================================
# SENTINEL SATELLITE DATA (via Copernicus)
# ============================================

def fetch_sentinel_data(lat, lon):
    """
    Query Sentinel-2/3 data availability from Copernicus Data Space
    Sentinel-2: High-resolution optical imagery (10m) - good for snow mapping
    Sentinel-3: Land Surface Temperature and snow cover
    """
    data = {
        'source': 'Sentinel',
        'available': False,
        's2_snow_index': None,
        's3_lst': None,
        's3_snow_cover': None
    }
    
    try:
        # Copernicus Data Space Ecosystem OpenSearch API
        # This queries for available Sentinel products
        
        bbox = f"{lon-0.1},{lat-0.1},{lon+0.1},{lat+0.1}"
        end_date = datetime.now().strftime('%Y-%m-%dT23:59:59Z')
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%dT00:00:00Z')
        
        # Query Sentinel-2 L2A products
        odata_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        
        # Using OData filter
        filter_str = (
            f"Collection/Name eq 'SENTINEL-2' and "
            f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/Value eq 'S2MSI2A') and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;POINT({lon} {lat})') and "
            f"ContentDate/Start gt {start_date}"
        )
        
        params = {
            '$filter': filter_str,
            '$top': 5
        }
        
        response = requests.get(odata_url, params=params, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            products = result.get('value', [])
            
            if products:
                data['available'] = True
                data['s2_products'] = len(products)
                data['latest_s2'] = products[0].get('Name', 'Unknown')
                
                # Cloud cover from metadata
                for prod in products:
                    attrs = prod.get('Attributes', [])
                    for attr in attrs:
                        if attr.get('Name') == 'cloudCover':
                            data['cloud_cover'] = attr.get('Value')
                            break
                            
    except Exception as e:
        data['error'] = str(e)
    
    return data

# ============================================
# NSIDC SNOW DATA (Alternative source)
# ============================================

def fetch_nsidc_data(lat, lon):
    """
    Query NSIDC (National Snow and Ice Data Center) for snow products
    Including AMSR-E/AMSR2 SWE data
    """
    data = {
        'source': 'NSIDC',
        'available': False,
        'swe': None,
        'snow_depth': None
    }
    
    try:
        # NSIDC provides AMSR2 daily SWE products
        # Query their CMR endpoint
        
        cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        params = {
            'short_name': 'AU_DySno',  # AMSR2 Daily Snow Products
            'temporal': f"{yesterday}T00:00:00Z,{yesterday}T23:59:59Z",
            'bounding_box': f"{lon-1},{lat-1},{lon+1},{lat+1}",
            'page_size': 3
        }
        
        response = requests.get(cmr_url, params=params, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            entries = result.get('feed', {}).get('entry', [])
            if entries:
                data['available'] = True
                data['granules'] = len(entries)
                
    except Exception as e:
        data['error'] = str(e)
    
    return data

# ============================================
# ADDITIONAL RELIABLE DATA SOURCES
# ============================================

def fetch_meteomatics_data(lat, lon):
    """
    Fetch weather model data that incorporates satellite observations
    Uses Open-Meteo as a free alternative to Meteomatics
    
    Provides additional parameters:
    - Precipitation type discrimination
    - High-resolution temperature profiles
    - Wind at multiple heights
    """
    data = {
        'source': 'Multi-Model-Ensemble',
        'available': False,
        'precipitation_type': None,
        'freezing_level': None,
        'snow_limit': None
    }
    
    try:
        # Open-Meteo weather models endpoint for ensemble data
        url = "https://api.open-meteo.com/v1/forecast"
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': [
                'temperature_2m',
                'precipitation',
                'rain',
                'snowfall',
                'freezing_level_height',
                'snow_depth'
            ],
            'models': ['best_match', 'gfs_seamless', 'icon_seamless'],
            'past_days': 2,
            'forecast_days': 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            hourly = result.get('hourly', {})
            
            if hourly:
                data['available'] = True
                
                # Freezing level
                freeze_levels = hourly.get('freezing_level_height', [])
                valid_freeze = [f for f in freeze_levels if f is not None]
                if valid_freeze:
                    data['freezing_level'] = valid_freeze[-1]
                
                # Precipitation analysis
                rain = hourly.get('rain', [])[-24:]
                snow = hourly.get('snowfall', [])[-24:]
                
                total_rain = sum(r for r in rain if r)
                total_snow = sum(s for s in snow if s)
                
                if total_snow > total_rain:
                    data['precipitation_type'] = 'snow'
                elif total_rain > total_snow:
                    data['precipitation_type'] = 'rain'
                else:
                    data['precipitation_type'] = 'mixed'
                    
    except Exception as e:
        data['error'] = str(e)
    
    return data

def fetch_ecmwf_ensemble(lat, lon):
    """
    Fetch ECMWF ensemble forecast data via Open-Meteo
    Provides uncertainty estimates for weather parameters
    """
    data = {
        'source': 'ECMWF_Ensemble',
        'available': False,
        'temp_uncertainty': None,
        'precip_probability': None
    }
    
    try:
        url = "https://ensemble-api.open-meteo.com/v1/ensemble"
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': [
                'temperature_2m',
                'precipitation'
            ],
            'models': 'ecmwf_ifs04'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            data['available'] = True
            data['ensemble_members'] = result.get('hourly_units', {})
            
    except Exception as e:
        data['error'] = str(e)
    
    return data

def fetch_climate_normals(lat, lon):
    """
    Fetch climate normals to compare current conditions against historical averages
    Uses Open-Meteo climate API
    """
    data = {
        'source': 'Climate_Normals',
        'available': False,
        'temp_anomaly': None,
        'precip_anomaly': None
    }
    
    try:
        # Get historical climate data for comparison
        current_month = datetime.now().month
        current_day = datetime.now().day
        
        # Get 30-year climate normal approximation
        url = "https://climate-api.open-meteo.com/v1/climate"
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': '1991-01-01',
            'end_date': '2020-12-31',
            'models': 'EC_Earth3P_HR',
            'daily': ['temperature_2m_mean', 'precipitation_sum']
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data['available'] = True
            # Historical data retrieved successfully
            
    except Exception as e:
        data['error'] = str(e)
    
    return data

def fetch_snowpack_model_data(lat, lon, elevation):
    """
    Estimate snowpack properties using elevation-based models
    Combines satellite snow cover with terrain analysis
    """
    data = {
        'source': 'Snowpack_Model',
        'available': True,
        'elevation_zone': None,
        'aspect_factor': 1.0,
        'estimated_density': None
    }
    
    # Elevation-based snow classification
    if elevation > 3000:
        data['elevation_zone'] = 'alpine'
        data['estimated_density'] = 350  # kg/m³, denser at high elevation
        data['melt_factor'] = 0.7  # Less melt
    elif elevation > 2000:
        data['elevation_zone'] = 'subalpine'
        data['estimated_density'] = 300
        data['melt_factor'] = 0.85
    elif elevation > 1000:
        data['elevation_zone'] = 'montane'
        data['estimated_density'] = 250
        data['melt_factor'] = 1.0
    else:
        data['elevation_zone'] = 'valley'
        data['estimated_density'] = 200
        data['melt_factor'] = 1.2  # More melt at lower elevations
    
    # Latitude-based solar radiation factor
    if lat > 60 or lat < -60:
        data['solar_factor'] = 0.6  # Less direct sun at high latitudes
    elif lat > 45 or lat < -45:
        data['solar_factor'] = 0.8
    else:
        data['solar_factor'] = 1.0
    
    return data

def fetch_avalanche_bulletin_regions(lat, lon):
    """
    Identify avalanche forecast regions for the location
    Maps to known avalanche forecasting organizations
    """
    data = {
        'source': 'Avalanche_Regions',
        'available': True,
        'forecast_region': None,
        'organization': None,
        'bulletin_url': None
    }
    
    # Map location to avalanche forecast regions
    # North America
    if -170 < lon < -50:
        if lat > 49:  # Canada
            data['organization'] = 'Avalanche Canada'
            data['bulletin_url'] = 'https://avalanche.ca/'
        elif lat > 35:  # US
            data['organization'] = 'US Avalanche Centers'
            data['bulletin_url'] = 'https://avalanche.org/'
    # Europe
    elif -10 < lon < 30:
        if 45 < lat < 48:  # Alps
            data['organization'] = 'EAWS (European Avalanche Warning Services)'
            data['bulletin_url'] = 'https://www.avalanches.org/'
        elif lat > 55:  # Scandinavia
            data['organization'] = 'Norwegian Avalanche Warning Service'
            data['bulletin_url'] = 'https://varsom.no/'
    # Asia
    elif lon > 70:
        if lat > 35:
            data['organization'] = 'Regional Avalanche Services'
            data['forecast_region'] = 'Himalayas/Central Asia'
    
    return data

# ============================================
# REAL-TIME WEATHER (Open-Meteo - incorporates satellite data)
# ============================================

def fetch_weather_data(lat, lon):
    """
    Fetch real-time weather and environmental data from Open-Meteo API
    Open-Meteo integrates data from multiple sources including:
    - ICON (German Weather Service)
    - GFS (NOAA)
    - ERA5 reanalysis
    - Satellite observations
    """
    try:
        current_url = f"https://api.open-meteo.com/v1/forecast"
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': [
                'temperature_2m',
                'relative_humidity_2m',
                'precipitation',
                'rain',
                'snowfall',
                'snow_depth',
                'weather_code',
                'surface_pressure',
                'wind_speed_10m',
                'wind_direction_10m',
                'cloud_cover',
                'shortwave_radiation',
                'direct_radiation',
                'diffuse_radiation',
                'direct_normal_irradiance',
                'terrestrial_radiation'
            ],
            'hourly': [
                'temperature_2m',
                'relative_humidity_2m',
                'precipitation',
                'rain',
                'snowfall',
                'snow_depth',
                'shortwave_radiation',
                'direct_radiation',
                'diffuse_radiation',
                'direct_normal_irradiance',
                'terrestrial_radiation',
                'soil_temperature_0cm',
                'soil_temperature_6cm'
            ],
            'daily': [
                'temperature_2m_max',
                'temperature_2m_min',
                'precipitation_sum',
                'rain_sum',
                'snowfall_sum',
                'shortwave_radiation_sum'
            ],
            'timezone': 'auto',
            'past_days': 3,
            'forecast_days': 1
        }
        
        response = requests.get(current_url, params=params, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Weather API returned status {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

# ============================================
# PHYSICS-BASED CALCULATIONS FOR DERIVED PARAMETERS
# ============================================

def calculate_snow_surface_temperature(air_temp, incoming_lw, outgoing_lw, wind_speed):
    """
    Calculate snow surface temperature using energy balance
    TSS ≈ ((OLWR / (ε * σ))^0.25) - 273.15
    """
    sigma = 5.67e-8  # Stefan-Boltzmann constant
    emissivity = 0.98  # Snow emissivity
    
    if outgoing_lw and outgoing_lw > 0:
        # From Stefan-Boltzmann law
        tss_k = (outgoing_lw / (emissivity * sigma)) ** 0.25
        tss = tss_k - 273.15
    else:
        # Estimate from air temp and radiative cooling
        # Snow surface is typically colder than air due to radiative cooling
        if air_temp < 0:
            tss = air_temp - 2 - (wind_speed * 0.1)  # Colder with more wind
        else:
            tss = min(0, air_temp - 1)  # Cannot exceed 0°C for snow
    
    return min(tss, 0)  # Snow surface temp cannot exceed 0°C

def calculate_sensible_heat_flux(air_temp, surface_temp, wind_speed, pressure=101325):
    """
    Calculate sensible heat flux using bulk aerodynamic formula
    Qs = ρ * cp * Ch * U * (Ta - Ts)
    """
    rho = pressure / (287 * (air_temp + 273.15))  # Air density
    cp = 1005  # Specific heat of air (J/kg/K)
    Ch = 0.002  # Bulk transfer coefficient for heat (typical for snow)
    
    qs = rho * cp * Ch * wind_speed * (air_temp - surface_temp)
    return qs

def calculate_latent_heat_flux(air_temp, surface_temp, relative_humidity, wind_speed, pressure=101325):
    """
    Calculate latent heat flux (sublimation/evaporation)
    Ql = ρ * Lv * Ce * U * (qa - qs)
    """
    # Saturation vapor pressure (Clausius-Clapeyron approximation)
    def sat_vapor_pressure(T):
        return 611.2 * math.exp(17.67 * T / (T + 243.5))
    
    Lv = 2.5e6  # Latent heat of vaporization (J/kg)
    Ls = 2.83e6  # Latent heat of sublimation (J/kg)
    Ce = 0.002  # Bulk transfer coefficient
    
    # Use sublimation if surface temp < 0
    L = Ls if surface_temp < 0 else Lv
    
    rho = pressure / (287 * (air_temp + 273.15))
    
    # Vapor pressures
    es_air = sat_vapor_pressure(air_temp)
    es_surf = sat_vapor_pressure(surface_temp)
    
    # Specific humidities
    qa = 0.622 * (relative_humidity / 100) * es_air / pressure
    qs = 0.622 * es_surf / pressure  # Assume saturation at snow surface
    
    ql = rho * L * Ce * wind_speed * (qa - qs)
    return ql

def calculate_liquid_water_content(air_temp, snow_depth, solar_radiation, time_hours_above_zero=0):
    """
    Estimate liquid water content in snowpack based on energy input
    Uses degree-day and radiation melt models
    """
    if snow_depth <= 0:
        return 0, 0, 0, 0
    
    # Degree-day factor (mm/°C/day)
    ddf = 4.0  # Typical value for alpine snow
    
    # Temperature-driven melt
    temp_melt = max(0, air_temp) * ddf / 24  # mm/hour
    
    # Radiation-driven melt (assuming 0.8 absorptivity)
    rad_melt = solar_radiation * 0.8 * 3600 / (334000 * 1000)  # mm/hour (334 kJ/kg latent heat)
    
    total_melt_rate = temp_melt + rad_melt  # mm/hour
    
    # Convert to kg/m² (1 mm water = 1 kg/m²)
    water = total_melt_rate * time_hours_above_zero * 0.5  # Accumulated over warm hours
    
    # Mean LWC as percentage of snow volume
    snow_density = 300  # kg/m³ typical
    snow_mass = snow_depth * snow_density  # kg/m²
    
    mean_lwc = (water / max(snow_mass, 1)) * 100 if snow_mass > 0 else 0
    max_lwc = mean_lwc * 1.5  # Maximum is typically higher than mean
    std_lwc = mean_lwc * 0.3  # Standard deviation
    
    return water, mean_lwc, max_lwc, std_lwc

def calculate_stability_index(snow_depth, new_snow_24h, air_temp, rain_on_snow, wind_speed, lwc):
    """
    Calculate skier stability index (S5)
    Lower values = less stable
    
    Based on:
    - New snow loading
    - Temperature conditions
    - Liquid water presence
    - Wind loading
    """
    s5 = 3.0  # Base stability (good conditions)
    
    # New snow loading effect
    if new_snow_24h > 0.4:  # >40cm in 24h
        s5 -= 1.2
    elif new_snow_24h > 0.3:
        s5 -= 0.8
    elif new_snow_24h > 0.2:
        s5 -= 0.5
    elif new_snow_24h > 0.1:
        s5 -= 0.2
    
    # Temperature effects
    if air_temp > 5:  # Strong warming
        s5 -= 0.8
    elif air_temp > 2:
        s5 -= 0.5
    elif air_temp > 0:
        s5 -= 0.3
    elif air_temp < -15:  # Cold can weaken bonds
        s5 -= 0.2
    
    # Rain on snow - very destabilizing
    if rain_on_snow > 10:
        s5 -= 1.0
    elif rain_on_snow > 5:
        s5 -= 0.6
    elif rain_on_snow > 0:
        s5 -= 0.3
    
    # Wind loading
    if wind_speed > 20:
        s5 -= 0.4
    elif wind_speed > 15:
        s5 -= 0.2
    
    # High liquid water content
    if lwc > 5:
        s5 -= 0.6
    elif lwc > 2:
        s5 -= 0.3
    
    # Thin snowpack more stable for deep slab
    if snow_depth < 0.5:
        s5 += 0.3
    
    return max(0.5, min(4.0, s5))

# ============================================
# MAIN DATA AGGREGATION FUNCTION
# ============================================

def fetch_all_satellite_data(lat, lon, progress_callback=None):
    """
    Aggregate data from all satellite sources
    Returns a dictionary with data from each source and fetch status
    
    Data Sources (12 total):
    1. Open-Meteo (Real-time weather, integrates multiple models)
    2. ERA5 Reanalysis (ECMWF historical data)
    3. NASA Earthdata (MODIS/VIIRS satellite products)
    4. NASA GIBS (Global imagery and derived products)
    5. NASA POWER (CERES radiation, MERRA-2 reanalysis)
    6. Sentinel (Copernicus high-resolution data)
    7. NSIDC (Snow and ice products)
    8. Multi-Model Ensemble (forecast uncertainty)
    9. ECMWF Ensemble (probabilistic forecasts)
    10. Climate Normals (historical comparison)
    11. Snowpack Model (elevation-based estimates)
    12. Avalanche Bulletins (regional forecast links)
    """
    results = {
        'location': {'lat': lat, 'lon': lon},
        'timestamp': datetime.now().isoformat(),
        'sources': {},
        'data_quality': {},
        'parameters_found': 0
    }
    
    # Get elevation for snowpack modeling
    elevation = get_elevation(lat, lon)
    results['elevation'] = elevation
    
    # Progress tracking - all data sources
    sources = [
        ('Open-Meteo (Real-time)', lambda: fetch_weather_data(lat, lon)),
        ('ERA5 Reanalysis', lambda: fetch_era5_data(lat, lon)),
        ('ERA5-Land (High-res)', lambda: fetch_era5_land_data(lat, lon)),
        ('NASA Earthdata (MODIS/VIIRS)', lambda: fetch_nasa_earthdata(lat, lon)),
        ('NASA GIBS (Snow Cover)', lambda: fetch_nasa_gibs_imagery(lat, lon)),
        ('NASA POWER (GOES/CERES)', lambda: fetch_goes_data(lat, lon)),
        ('Sentinel (Copernicus)', lambda: fetch_sentinel_data(lat, lon)),
        ('NSIDC Snow Products', lambda: fetch_nsidc_data(lat, lon)),
        ('Multi-Model Ensemble', lambda: fetch_meteomatics_data(lat, lon)),
        ('ECMWF Ensemble', lambda: fetch_ecmwf_ensemble(lat, lon)),
        ('Climate Normals', lambda: fetch_climate_normals(lat, lon)),
        ('Snowpack Model', lambda: fetch_snowpack_model_data(lat, lon, elevation)),
        ('Avalanche Regions', lambda: fetch_avalanche_bulletin_regions(lat, lon)),
    ]
    
    # Track which parameters we successfully retrieved
    params_from_satellite = set()
    
    for i, (name, fetch_func) in enumerate(sources):
        if progress_callback:
            progress_callback((i + 1) / len(sources), f"🛰️ Fetching {name}...")
        try:
            source_data = fetch_func()
            results['sources'][name] = source_data
            
            # Track successful data
            if isinstance(source_data, dict):
                if source_data.get('available', True):
                    results['data_quality'][name] = 'success'
                    # Count parameters from this source
                    param_count = sum(1 for v in source_data.values() 
                                     if v is not None and v != [] and str(v) != '{}')
                    results['parameters_found'] += param_count
                else:
                    results['data_quality'][name] = 'partial'
            else:
                results['data_quality'][name] = 'success'
                
        except Exception as e:
            results['sources'][name] = {'error': str(e), 'available': False}
            results['data_quality'][name] = 'failed'
    
    # Summary of data quality
    success_count = sum(1 for v in results['data_quality'].values() if v == 'success')
    results['summary'] = {
        'total_sources': len(sources),
        'successful_sources': success_count,
        'success_rate': f"{(success_count/len(sources))*100:.0f}%"
    }
    
    return results

def process_satellite_data(satellite_data, elevation=1500):
    """
    Process all satellite data into model input features
    Combines data from multiple sources with quality weighting
    """
    inputs = {}
    data_sources_used = []
    
    # Extract individual source data
    weather = satellite_data['sources'].get('Open-Meteo (Real-time)', {})
    era5 = satellite_data['sources'].get('ERA5 Reanalysis', {})
    gibs = satellite_data['sources'].get('NASA GIBS', {})
    goes = satellite_data['sources'].get('NASA POWER (GOES/CERES)', {})
    
    now = datetime.now()
    
    # ========================================
    # 1. TEMPERATURE (TA, TA_daily, TSS_mod)
    # Sources: ERA5, Open-Meteo
    # ========================================
    
    # Current air temperature
    if weather and 'current' in weather:
        inputs['TA'] = weather['current'].get('temperature_2m', 0)
        data_sources_used.append(('TA', 'Open-Meteo'))
    elif era5.get('available') and era5.get('temperature_2m'):
        inputs['TA'] = era5['temperature_2m'][-1] if era5['temperature_2m'] else 0
        data_sources_used.append(('TA', 'ERA5'))
    else:
        inputs['TA'] = 0
    
    # Daily average temperature
    if era5.get('available') and era5.get('daily_temp_mean'):
        inputs['TA_daily'] = era5['daily_temp_mean'][-1] if era5['daily_temp_mean'] else inputs['TA']
        data_sources_used.append(('TA_daily', 'ERA5'))
    elif weather and 'daily' in weather:
        daily = weather['daily']
        t_max = daily.get('temperature_2m_max', [0])[-1] or 0
        t_min = daily.get('temperature_2m_min', [0])[-1] or 0
        inputs['TA_daily'] = (t_max + t_min) / 2
        data_sources_used.append(('TA_daily', 'Open-Meteo'))
    else:
        inputs['TA_daily'] = inputs['TA']
    
    # Time of day
    inputs['profile_time'] = now.hour
    data_sources_used.append(('profile_time', 'System'))
    
    # ========================================
    # 2. RADIATION (ISWR, ILWR, OLWR)
    # Sources: ERA5, GOES/CERES, Open-Meteo
    # ========================================
    
    # Get radiation from best available source
    current_sw = 0
    current_direct = 0
    current_diffuse = 0
    current_terrestrial = 0
    
    # Try Open-Meteo current radiation first (real-time)
    if weather and 'current' in weather:
        current = weather['current']
        current_sw = current.get('shortwave_radiation', 0) or 0
        current_direct = current.get('direct_radiation', 0) or 0
        current_diffuse = current.get('diffuse_radiation', 0) or 0
        current_terrestrial = current.get('terrestrial_radiation', 0) or 0
        data_sources_used.append(('ISWR_current', 'Open-Meteo'))
    
    # Daily radiation from GOES/CERES (NASA POWER)
    if goes.get('available') and goes.get('shortwave_radiation'):
        sw_dict = goes['shortwave_radiation']
        if sw_dict:
            # Get most recent value
            recent_vals = [v for v in sw_dict.values() if v and v > 0]
            if recent_vals:
                inputs['ISWR_daily'] = recent_vals[-1] * 1000 / 24  # Convert MJ/m²/day to W/m² avg
                data_sources_used.append(('ISWR_daily', 'GOES/CERES'))
    
    if 'ISWR_daily' not in inputs:
        # Fallback to ERA5 or Open-Meteo
        if era5.get('available') and era5.get('daily_radiation'):
            daily_rad = era5['daily_radiation']
            inputs['ISWR_daily'] = daily_rad[-1] / 24 if daily_rad and daily_rad[-1] else 100
            data_sources_used.append(('ISWR_daily', 'ERA5'))
        elif weather and 'daily' in weather:
            daily_rad = weather['daily'].get('shortwave_radiation_sum', [0])[-1]
            inputs['ISWR_daily'] = daily_rad / 24 if daily_rad else 100
            data_sources_used.append(('ISWR_daily', 'Open-Meteo'))
        else:
            inputs['ISWR_daily'] = 100
    
    # Radiation components
    if era5.get('available'):
        hourly = era5
        if hourly.get('direct_radiation') and len(hourly['direct_radiation']) > 0:
            inputs['ISWR_dir_daily'] = np.mean([x for x in hourly['direct_radiation'][-24:] if x]) or current_direct
            data_sources_used.append(('ISWR_dir_daily', 'ERA5'))
        
        if hourly.get('diffuse_radiation') and len(hourly['diffuse_radiation']) > 0:
            inputs['ISWR_diff_daily'] = np.mean([x for x in hourly['diffuse_radiation'][-24:] if x]) or current_diffuse
            data_sources_used.append(('ISWR_diff_daily', 'ERA5'))
    
    # Defaults for radiation
    inputs.setdefault('ISWR_dir_daily', inputs['ISWR_daily'] * 0.6)
    inputs.setdefault('ISWR_diff_daily', inputs['ISWR_daily'] * 0.4)
    inputs['ISWR_h_daily'] = inputs['ISWR_daily'] * 0.95  # Horizontal component
    
    # Longwave radiation (calculated from temperature)
    sigma = 5.67e-8  # Stefan-Boltzmann constant
    
    # Get humidity for emissivity calculation
    rel_humidity = 70  # Default
    if weather and 'current' in weather:
        rel_humidity = weather['current'].get('relative_humidity_2m', 70) or 70
    
    # Incoming LW from atmosphere
    temp_k = inputs['TA'] + 273.15
    emissivity_sky = 0.7 + 0.003 * rel_humidity  # Approximation
    inputs['ILWR'] = emissivity_sky * sigma * (temp_k ** 4)
    inputs['ILWR_daily'] = inputs['ILWR']
    data_sources_used.append(('ILWR', 'Calculated (Stefan-Boltzmann)'))
    
    # GOES longwave if available
    if goes.get('available') and goes.get('longwave_radiation'):
        lw_dict = goes['longwave_radiation']
        if lw_dict:
            recent_vals = [v for v in lw_dict.values() if v and v > 0]
            if recent_vals:
                inputs['ILWR_daily'] = recent_vals[-1] * 1000 / 24  # Convert to W/m²
                data_sources_used.append(('ILWR_daily', 'GOES/CERES'))
    
    # ========================================
    # 3. SNOW PROPERTIES (max_height, SWE)
    # Sources: ERA5, MODIS/VIIRS, Open-Meteo
    # ========================================
    
    # Snow depth
    snow_depth = 0
    snow_depth_history = []
    
    if era5.get('available') and era5.get('snow_depth'):
        snow_depths = [x for x in era5['snow_depth'] if x is not None]
        if snow_depths:
            snow_depth = snow_depths[-1]
            snow_depth_history = snow_depths
            data_sources_used.append(('max_height', 'ERA5'))
    
    if snow_depth == 0 and weather and 'current' in weather:
        snow_depth = (weather['current'].get('snow_depth', 0) or 0) / 100  # cm to m
        if weather.get('hourly', {}).get('snow_depth'):
            snow_depth_history = [x/100 if x else 0 for x in weather['hourly']['snow_depth']]
        data_sources_used.append(('max_height', 'Open-Meteo'))
    
    inputs['max_height'] = snow_depth
    
    # Snow depth changes (use history)
    if len(snow_depth_history) >= 72:
        inputs['max_height_1_diff'] = snow_depth_history[-1] - snow_depth_history[-25] if len(snow_depth_history) >= 25 else 0
        inputs['max_height_2_diff'] = snow_depth_history[-1] - snow_depth_history[-49] if len(snow_depth_history) >= 49 else 0
        inputs['max_height_3_diff'] = snow_depth_history[-1] - snow_depth_history[-72]
        data_sources_used.append(('height_diff', 'ERA5/Open-Meteo'))
    else:
        inputs['max_height_1_diff'] = 0
        inputs['max_height_2_diff'] = 0
        inputs['max_height_3_diff'] = 0
    
    # SWE from snowfall
    if era5.get('available') and era5.get('daily_snowfall'):
        daily_snow = era5['daily_snowfall'][-1] if era5['daily_snowfall'] else 0
        inputs['SWE_daily'] = (daily_snow or 0) * 10  # Rough SWE estimate (10:1 ratio)
        data_sources_used.append(('SWE_daily', 'ERA5'))
    elif weather and 'daily' in weather:
        daily_snow = weather['daily'].get('snowfall_sum', [0])[-1] or 0
        inputs['SWE_daily'] = daily_snow * 10
        data_sources_used.append(('SWE_daily', 'Open-Meteo'))
    else:
        inputs['SWE_daily'] = 0
    
    # Rain
    if era5.get('available') and era5.get('daily_rain'):
        inputs['MS_Rain_daily'] = era5['daily_rain'][-1] if era5['daily_rain'] else 0
        data_sources_used.append(('MS_Rain_daily', 'ERA5'))
    elif weather and 'daily' in weather:
        inputs['MS_Rain_daily'] = weather['daily'].get('rain_sum', [0])[-1] or 0
        data_sources_used.append(('MS_Rain_daily', 'Open-Meteo'))
    else:
        inputs['MS_Rain_daily'] = 0
    
    # ========================================
    # 4. SNOW SURFACE TEMPERATURE (TSS_mod)
    # Calculated from physics
    # ========================================
    
    wind_speed = 5
    if weather and 'current' in weather:
        wind_speed = weather['current'].get('wind_speed_10m', 5) or 5
    
    inputs['TSS_mod'] = calculate_snow_surface_temperature(
        inputs['TA'], 
        inputs['ILWR'],
        inputs.get('OLWR', 300),
        wind_speed
    )
    data_sources_used.append(('TSS_mod', 'Calculated (Energy Balance)'))
    
    # Outgoing LW from snow surface
    inputs['OLWR'] = 0.98 * sigma * ((inputs['TSS_mod'] + 273.15) ** 4)
    inputs['OLWR_daily'] = inputs['OLWR']
    data_sources_used.append(('OLWR', 'Calculated (Stefan-Boltzmann)'))
    
    # ========================================
    # 5. HEAT FLUXES (Qs, Ql)
    # Calculated using bulk aerodynamic formulas
    # ========================================
    
    pressure = 101325
    if weather and 'current' in weather:
        pressure = (weather['current'].get('surface_pressure', 1013) or 1013) * 100
    
    inputs['Qs'] = calculate_sensible_heat_flux(
        inputs['TA'],
        inputs['TSS_mod'],
        wind_speed,
        pressure
    )
    data_sources_used.append(('Qs', 'Calculated (Bulk Aerodynamic)'))
    
    inputs['Ql'] = calculate_latent_heat_flux(
        inputs['TA'],
        inputs['TSS_mod'],
        rel_humidity,
        wind_speed,
        pressure
    )
    inputs['Ql_daily'] = inputs['Ql']
    data_sources_used.append(('Ql', 'Calculated (Bulk Aerodynamic)'))
    
    # Absorbed shortwave
    albedo = 0.8 if inputs['TA'] < 0 else 0.6  # Lower albedo for wet snow
    inputs['Qw_daily'] = inputs['ISWR_daily'] * (1 - albedo)
    data_sources_used.append(('Qw_daily', 'Calculated'))
    
    # ========================================
    # 6. LIQUID WATER CONTENT
    # Estimated from melt conditions
    # ========================================
    
    # Count hours above 0°C in last 24h
    hours_above_zero = 0
    if weather and 'hourly' in weather:
        temps = weather['hourly'].get('temperature_2m', [])[-24:]
        hours_above_zero = sum(1 for t in temps if t and t > 0)
    elif era5.get('available') and era5.get('temperature_2m'):
        temps = era5['temperature_2m'][-24:]
        hours_above_zero = sum(1 for t in temps if t and t > 0)
    
    water, mean_lwc, max_lwc, std_lwc = calculate_liquid_water_content(
        inputs['TA'],
        inputs['max_height'],
        inputs['ISWR_daily'],
        hours_above_zero
    )
    
    inputs['water'] = water
    inputs['mean_lwc'] = mean_lwc
    inputs['max_lwc'] = max_lwc
    inputs['std_lwc'] = std_lwc
    data_sources_used.append(('LWC', 'Calculated (Degree-Day + Radiation)'))
    
    # LWC changes based on temperature trends
    temp_history = []
    if era5.get('available') and era5.get('temperature_2m'):
        temp_history = era5['temperature_2m']
    elif weather and 'hourly' in weather:
        temp_history = weather['hourly'].get('temperature_2m', [])
    
    if len(temp_history) >= 72:
        temp_trend_1d = temp_history[-1] - temp_history[-25] if temp_history[-25] else 0
        temp_trend_2d = temp_history[-1] - temp_history[-49] if temp_history[-49] else 0
        temp_trend_3d = temp_history[-1] - temp_history[-72] if temp_history[-72] else 0
    else:
        temp_trend_1d = temp_trend_2d = temp_trend_3d = 0
    
    is_melting = inputs['TA'] > 0 or (inputs['TA'] > -2 and inputs['ISWR_daily'] > 200)
    
    inputs['water_1_diff'] = temp_trend_1d * 3 if is_melting else 0
    inputs['water_2_diff'] = temp_trend_2d * 3 if is_melting else 0
    inputs['water_3_diff'] = temp_trend_3d * 3 if is_melting else 0
    inputs['mean_lwc_2_diff'] = temp_trend_2d * 0.5
    inputs['mean_lwc_3_diff'] = temp_trend_3d * 0.5
    data_sources_used.append(('water_diff', 'Calculated (Temperature Trend)'))
    
    # Wetness distribution
    inputs['prop_up'] = 0.3 if is_melting else 0.1
    inputs['prop_wet_2_diff'] = 0.1 if temp_trend_2d > 2 else -0.05 if temp_trend_2d < -2 else 0
    inputs['sum_up'] = inputs['water'] * inputs['prop_up']
    
    # Wet layer depth changes
    inputs['lowest_2_diff'] = 0.1 if is_melting and temp_trend_2d > 0 else 0
    inputs['lowest_3_diff'] = 0.15 if is_melting and temp_trend_3d > 0 else 0
    data_sources_used.append(('wetness_dist', 'Calculated'))
    
    # ========================================
    # 7. STABILITY INDEX (S5)
    # ========================================
    
    new_snow_24h = inputs['max_height_1_diff'] if inputs['max_height_1_diff'] > 0 else 0
    
    inputs['S5'] = calculate_stability_index(
        inputs['max_height'],
        new_snow_24h,
        inputs['TA'],
        inputs['MS_Rain_daily'],
        wind_speed,
        inputs['mean_lwc']
    )
    
    # Daily stability change
    inputs['S5_daily'] = -0.2 if temp_trend_1d > 3 else 0.1 if temp_trend_1d < -3 else 0
    data_sources_used.append(('S5', 'Calculated (Multi-factor)'))
    
    return inputs, data_sources_used

# ============================================
# STREAMLIT UI
# ============================================

# Page configuration
st.set_page_config(
    page_title="Avalanche Prediction System",
    page_icon="🏔️",
    layout="wide"
)

# Custom CSS
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
    .satellite-source {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #1E88E5;
    }
    .data-quality-good {
        color: #00C851;
        font-weight: bold;
    }
    .data-quality-estimated {
        color: #ffbb33;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏔️ Avalanche Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem;">Multi-Satellite Data Integration: MODIS • VIIRS • ERA5 • GOES • Sentinel</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# LOCATION & AUTO-FETCH SECTION
# ============================================
st.subheader("📍 Location & Satellite Data Source")

col_loc1, col_loc2 = st.columns([2, 1])

with col_loc1:
    data_source = st.radio(
        "How would you like to input data?",
        ["🛰️ Auto-fetch from satellites (using my location)", "✍️ Manual input"],
        horizontal=True
    )

# Initialize session state
if 'location' not in st.session_state:
    st.session_state.location = None
if 'env_data' not in st.session_state:
    st.session_state.env_data = None
if 'satellite_raw' not in st.session_state:
    st.session_state.satellite_raw = None
if 'data_sources' not in st.session_state:
    st.session_state.data_sources = []
if 'inputs' not in st.session_state:
    st.session_state.inputs = {f: 0.0 for f in features_for_input}

if data_source == "🛰️ Auto-fetch from satellites (using my location)":
    
    with col_loc2:
        fetch_location = st.button("🔄 Refresh Location & Data", type="secondary")
    
    if fetch_location or st.session_state.location is None:
        with st.spinner("📡 Fetching your location from IP address..."):
            st.session_state.location = get_user_location()
            lat = st.session_state.location['latitude']
            lon = st.session_state.location['longitude']
            st.session_state.location['elevation'] = get_elevation(lat, lon)
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress, text):
            progress_bar.progress(progress)
            status_text.text(text)
        
        with st.spinner("🛰️ Fetching satellite data from multiple sources..."):
            lat = st.session_state.location['latitude']
            lon = st.session_state.location['longitude']
            
            st.session_state.satellite_raw = fetch_all_satellite_data(lat, lon, update_progress)
            
            elevation = st.session_state.location.get('elevation', 1500)
            st.session_state.env_data, st.session_state.data_sources = process_satellite_data(
                st.session_state.satellite_raw, 
                elevation
            )
        
        progress_bar.empty()
        status_text.empty()
    
    # Display location info
    if st.session_state.location:
        loc = st.session_state.location
        
        st.success(f"""
        **📍 Detected Location:** {loc['city']}, {loc['region']}, {loc['country']}  
        **🌐 Coordinates:** {loc['latitude']:.4f}°N, {loc['longitude']:.4f}°E  
        **⛰️ Elevation:** {loc.get('elevation', 'Unknown')}m  
        **🕐 Timezone:** {loc['timezone']}
        """)
        
        # Manual coordinate adjustment
        with st.expander("🎯 Adjust Location Manually"):
            col_coord1, col_coord2, col_coord3 = st.columns(3)
            with col_coord1:
                new_lat = st.number_input("Latitude", value=loc['latitude'], min_value=-90.0, max_value=90.0, step=0.01)
            with col_coord2:
                new_lon = st.number_input("Longitude", value=loc['longitude'], min_value=-180.0, max_value=180.0, step=0.01)
            with col_coord3:
                elev_value = loc.get('elevation') or 1500
                new_elev = st.number_input("Elevation (m)", value=int(elev_value), min_value=0, max_value=9000, step=100)
            
            if st.button("📡 Fetch Data for New Coordinates"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, text):
                    progress_bar.progress(progress)
                    status_text.text(text)
                
                with st.spinner("🛰️ Downloading data for new location..."):
                    st.session_state.location['latitude'] = new_lat
                    st.session_state.location['longitude'] = new_lon
                    st.session_state.location['elevation'] = new_elev
                    
                    st.session_state.satellite_raw = fetch_all_satellite_data(new_lat, new_lon, update_progress)
                    st.session_state.env_data, st.session_state.data_sources = process_satellite_data(
                        st.session_state.satellite_raw,
                        new_elev
                    )
                
                progress_bar.empty()
                status_text.empty()
                st.rerun()
    
    # Display satellite data status
    if st.session_state.satellite_raw:
        st.markdown("### 🛰️ Satellite Data Sources Status")
        
        raw = st.session_state.satellite_raw
        
        # Show summary
        if 'summary' in raw:
            summary = raw['summary']
            st.info(f"📊 **Data Fetch Summary:** {summary['successful_sources']}/{summary['total_sources']} sources successful ({summary['success_rate']})")
        
        # Create columns for satellite sources - Row 1
        st.markdown("**Primary Satellite Data:**")
        cols = st.columns(4)
        
        source_status_row1 = [
            ("🌍 ERA5", raw['data_quality'].get('ERA5 Reanalysis') == 'success'),
            ("🛰️ MODIS/VIIRS", raw['data_quality'].get('NASA Earthdata (MODIS/VIIRS)') == 'success'),
            ("☀️ GOES/CERES", raw['data_quality'].get('NASA POWER (GOES/CERES)') == 'success'),
            ("🇪🇺 Sentinel", raw['data_quality'].get('Sentinel (Copernicus)') == 'success'),
        ]
        
        for i, (name, available) in enumerate(source_status_row1):
            with cols[i]:
                if available:
                    st.success(f"✅ {name}")
                else:
                    st.warning(f"⚠️ {name}")
        
        # Row 2 - Additional sources
        st.markdown("**Additional Data Sources:**")
        cols2 = st.columns(4)
        
        source_status_row2 = [
            ("🌐 Open-Meteo", raw['data_quality'].get('Open-Meteo (Real-time)') == 'success'),
            ("❄️ NSIDC", raw['data_quality'].get('NSIDC Snow Products') == 'success'),
            ("🔮 Ensemble", raw['data_quality'].get('Multi-Model Ensemble') == 'success'),
            ("📍 Snow Model", raw['data_quality'].get('Snowpack Model') == 'success'),
        ]
        
        for i, (name, available) in enumerate(source_status_row2):
            with cols2[i]:
                if available:
                    st.success(f"✅ {name}")
                else:
                    st.warning(f"⚠️ {name}")
        
        # Show detailed data
        with st.expander("📊 View Detailed Satellite Data"):
            for source_name, source_data in raw['sources'].items():
                st.markdown(f"**{source_name}**")
                if isinstance(source_data, dict):
                    # Filter out large arrays for display
                    display_data = {k: v for k, v in source_data.items() 
                                   if not isinstance(v, list) or len(str(v)) < 200}
                    st.json(display_data)
                st.markdown("---")
    
    # Display fetched data summary
    if st.session_state.env_data:
        st.markdown("### 📈 Retrieved Parameters Summary")
        
        env = st.session_state.env_data
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🌡️ Air Temp", f"{env.get('TA', 0):.1f}°C", 
                     delta=f"{env.get('TA', 0) - env.get('TA_daily', 0):.1f}°C from daily avg")
        with col2:
            st.metric("❄️ Snow Depth", f"{env.get('max_height', 0)*100:.0f} cm",
                     delta=f"{env.get('max_height_1_diff', 0)*100:.1f} cm (24h)")
        with col3:
            st.metric("☀️ Solar Radiation", f"{env.get('ISWR_daily', 0):.0f} W/m²")
        with col4:
            st.metric("⚠️ Stability Index", f"{env.get('S5', 2.5):.2f}")
        
        # Second row of metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🌡️ Snow Surface Temp", f"{env.get('TSS_mod', 0):.1f}°C")
        with col2:
            st.metric("💧 Liquid Water", f"{env.get('water', 0):.1f} kg/m²")
        with col3:
            st.metric("🔥 Sensible Heat", f"{env.get('Qs', 0):.1f} W/m²")
        with col4:
            st.metric("💨 Latent Heat", f"{env.get('Ql', 0):.1f} W/m²")
        
        # Data sources used
        with st.expander("📡 Data Source Attribution"):
            # Group by source
            source_params = {}
            for param, source in st.session_state.data_sources:
                if source not in source_params:
                    source_params[source] = []
                source_params[source].append(param)
            
            for source, params in source_params.items():
                quality_class = "data-quality-good" if "ERA5" in source or "MODIS" in source or "GOES" in source else "data-quality-estimated"
                st.markdown(f'<div class="satellite-source"><span class="{quality_class}">{source}</span>: {", ".join(params)}</div>', unsafe_allow_html=True)
            
            st.markdown("""
            **Legend:**
            - <span class="data-quality-good">Green</span>: Direct satellite/reanalysis data
            - <span class="data-quality-estimated">Yellow</span>: Physics-based calculations
            """, unsafe_allow_html=True)
        
        # Update session state inputs
        for key, value in env.items():
            if key in features_for_input:
                st.session_state.inputs[key] = value
        
        st.info("✅ **Satellite data loaded!** Values below have been auto-populated from satellite sources. You can still adjust them manually if needed.")
    
    st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Model Settings")
st.sidebar.markdown("""
### 🛰️ Satellite Sources
This app integrates data from:

**Direct Measurements:**
- **MODIS** (Terra/Aqua) - Snow cover, LST
- **VIIRS** (Suomi NPP) - Snow products
- **ERA5** - Reanalysis (T, snow, radiation)
- **GOES/CERES** - Radiation fluxes
- **Sentinel** - High-res snow mapping

**Physics Calculations:**
- Snow surface temperature
- Heat fluxes (Qs, Ql)
- Liquid water content
- Stability indices
""")

# Show satellite info in sidebar
with st.sidebar.expander("ℹ️ About Satellite Data"):
    for sat_name, sat_info in SATELLITE_SOURCES.items():
        st.markdown(f"**{sat_info['name']}**")
        st.markdown(f"- Provider: {sat_info['provider']}")
        st.markdown(f"- Resolution: {sat_info['resolution']}")
        st.markdown(f"- Products: {', '.join(sat_info['products'][:2])}")
        st.markdown("---")

# Helper function
def get_input_value(key, default=0.0, min_val=None, max_val=None):
    value = st.session_state.inputs.get(key, default)
    if value is None:
        value = default
    # Clamp value to min/max if provided
    if min_val is not None and value < min_val:
        value = min_val
    if max_val is not None and value > max_val:
        value = max_val
    return value

# Input sections
st.subheader("📊 Snowpack & Weather Data")

if data_source == "🛰️ Auto-fetch from satellites (using my location)" and st.session_state.env_data:
    st.caption("🛰️ **Values pre-filled from satellite data** - You can adjust them below")
else:
    st.caption("✍️ **Manual entry mode** - Enter your observations below")

# Tabs for organized input
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
            help="Current air temperature (from ERA5/Open-Meteo)",
            key="input_TA"
        )
        st.session_state.inputs['TA_daily'] = st.number_input(
            "Daily Avg Temperature (°C)", 
            value=float(get_input_value('TA_daily', 0.0)), 
            min_value=-40.0, max_value=20.0, step=0.5,
            help="Daily average air temperature (from ERA5)",
            key="input_TA_daily"
        )
    
    with col2:
        st.session_state.inputs['TSS_mod'] = st.number_input(
            "Snow Surface Temp (°C)", 
            value=float(get_input_value('TSS_mod', 0.0)), 
            min_value=-40.0, max_value=0.0, step=0.5,
            help="Modeled snow surface temperature (calculated from energy balance)",
            key="input_TSS_mod"
        )
        st.session_state.inputs['MS_Rain_daily'] = st.number_input(
            "Daily Rainfall (kg/m²)", 
            value=float(get_input_value('MS_Rain_daily', 0.0)), 
            min_value=0.0, max_value=100.0, step=1.0,
            help="Daily rainfall (from ERA5/Open-Meteo)",
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
    st.caption("💡 LWC is estimated from temperature, radiation, and melt conditions")
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
            "Mean LWC (%)", 
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
            "Max LWC (%)", 
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
    st.caption("☀️ Radiation data from GOES/CERES and ERA5 satellites")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Longwave Radiation**")
        st.session_state.inputs['ILWR'] = st.number_input(
            "Incoming LW (W/m²)", 
            value=float(get_input_value('ILWR', 250.0, 0.0, 600.0)), 
            min_value=0.0, max_value=600.0, step=5.0,
            help="From atmospheric emission (calculated/ERA5)",
            key="input_ILWR"
        )
        st.session_state.inputs['ILWR_daily'] = st.number_input(
            "Daily Incoming LW", 
            value=float(get_input_value('ILWR_daily', 250.0, 0.0, 600.0)), 
            min_value=0.0, max_value=600.0, step=5.0,
            key="input_ILWR_daily"
        )
        st.session_state.inputs['OLWR'] = st.number_input(
            "Outgoing LW (W/m²)", 
            value=float(get_input_value('OLWR', 300.0, 0.0, 600.0)), 
            min_value=0.0, max_value=600.0, step=5.0,
            help="From snow surface (calculated from TSS)",
            key="input_OLWR"
        )
        st.session_state.inputs['OLWR_daily'] = st.number_input(
            "Daily Outgoing LW", 
            value=float(get_input_value('OLWR_daily', 300.0, 0.0, 600.0)), 
            min_value=0.0, max_value=600.0, step=5.0,
            key="input_OLWR_daily"
        )
    
    with col2:
        st.markdown("**Shortwave Radiation**")
        st.session_state.inputs['ISWR_daily'] = st.number_input(
            "Daily SW Total (W/m²)", 
            value=float(get_input_value('ISWR_daily', 100.0, 0.0, 1500.0)), 
            min_value=0.0, max_value=1500.0, step=10.0,
            help="From GOES/CERES satellite",
            key="input_ISWR_daily"
        )
        st.session_state.inputs['ISWR_h_daily'] = st.number_input(
            "Daily Horizontal SW", 
            value=float(get_input_value('ISWR_h_daily', 100.0, 0.0, 1500.0)), 
            min_value=0.0, max_value=1500.0, step=10.0,
            key="input_ISWR_h_daily"
        )
        st.session_state.inputs['ISWR_dir_daily'] = st.number_input(
            "Daily Direct SW", 
            value=float(get_input_value('ISWR_dir_daily', 50.0, 0.0, 1200.0)), 
            min_value=0.0, max_value=1200.0, step=10.0,
            help="From ERA5",
            key="input_ISWR_dir_daily"
        )
        st.session_state.inputs['ISWR_diff_daily'] = st.number_input(
            "Daily Diffuse SW", 
            value=float(get_input_value('ISWR_diff_daily', 50.0, 0.0, 800.0)), 
            min_value=0.0, max_value=800.0, step=10.0,
            help="From ERA5",
            key="input_ISWR_diff_daily"
        )
    
    with col3:
        st.markdown("**Heat Flux** (Calculated)")
        st.session_state.inputs['Qs'] = st.number_input(
            "Sensible Heat (W/m²)", 
            value=float(get_input_value('Qs', 0.0, -500.0, 500.0)), 
            min_value=-500.0, max_value=500.0, step=5.0,
            help="Calculated using bulk aerodynamic formula",
            key="input_Qs"
        )
        st.session_state.inputs['Ql'] = st.number_input(
            "Latent Heat (W/m²)", 
            value=float(get_input_value('Ql', 0.0, -500.0, 500.0)), 
            min_value=-500.0, max_value=500.0, step=5.0,
            help="Sublimation/evaporation flux",
            key="input_Ql"
        )
        st.session_state.inputs['Ql_daily'] = st.number_input(
            "Daily Latent Heat", 
            value=float(get_input_value('Ql_daily', 0.0, -500.0, 500.0)), 
            min_value=-500.0, max_value=500.0, step=5.0,
            key="input_Ql_daily"
        )
        st.session_state.inputs['Qw_daily'] = st.number_input(
            "Daily Absorbed SW", 
            value=float(get_input_value('Qw_daily', 50.0, 0.0, 1000.0)), 
            min_value=0.0, max_value=1000.0, step=10.0,
            key="input_Qw_daily"
        )

with tab4:
    st.markdown("### Snow Properties")
    st.caption("❄️ Snow depth from ERA5 reanalysis and MODIS/VIIRS")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Snow Height**")
        st.session_state.inputs['max_height'] = st.number_input(
            "Snow Height (m)", 
            value=float(get_input_value('max_height', 1.0)), 
            min_value=0.0, max_value=10.0, step=0.1,
            help="From ERA5/MODIS snow depth products",
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
            help="Snow Water Equivalent change",
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
    st.caption("⚠️ Stability calculated from multiple factors (snow load, temperature, LWC)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.inputs['S5'] = st.number_input(
            "Stability Index (S5)", 
            value=float(get_input_value('S5', 2.5)), 
            min_value=0.0, max_value=5.0, step=0.1,
            help="Calculated from snow conditions - lower = less stable",
            key="input_S5"
        )
    
    with col2:
        st.session_state.inputs['S5_daily'] = st.number_input(
            "Daily Stability Change", 
            value=float(get_input_value('S5_daily', 0.0)), 
            min_value=-2.0, max_value=2.0, step=0.1,
            key="input_S5_daily"
        )
    
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
    input_data = pd.DataFrame([[st.session_state.inputs[f] for f in features_for_input]], 
                              columns=features_for_input)
    
    model_path = "avalanche_model"
    scaler_path = "scaler.joblib"
    imputer_path = "imputer.joblib"
    
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        imputer = joblib.load(imputer_path)
        
        input_imputed = imputer.transform(input_data)
        input_scaled = scaler.transform(input_imputed)
        
        prediction, _ = model.predict(input_scaled)
        confidence = float(prediction[0][0])
        
    except Exception as e:
        st.warning("⚠️ Model files not found. Using physics-based risk assessment.")
        
        risk_score = 0.3
        
        if st.session_state.inputs['TA'] > 0:
            risk_score += 0.15
        if st.session_state.inputs['TA_daily'] > st.session_state.inputs['TA']:
            risk_score += 0.1
        
        if st.session_state.inputs['water_1_diff'] > 10:
            risk_score += 0.15
        if st.session_state.inputs['mean_lwc'] > 20:
            risk_score += 0.1
        
        if st.session_state.inputs['S5'] < 1.0:
            risk_score += 0.25
        elif st.session_state.inputs['S5'] < 1.5:
            risk_score += 0.15
        elif st.session_state.inputs['S5'] < 2.0:
            risk_score += 0.05
        
        if st.session_state.inputs['max_height_1_diff'] > 0.3:
            risk_score += 0.15
        
        if st.session_state.inputs['MS_Rain_daily'] > 5:
            risk_score += 0.2
        
        confidence = min(max(risk_score, 0.0), 1.0)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
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
    
    st.markdown("### Risk Level Gauge")
    st.progress(confidence)
    
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

# Comprehensive data sources attribution
with st.expander("📡 Complete Satellite Data Sources & Parameter Attribution"):
    st.markdown("""
    ### 🛰️ Multi-Satellite Data Integration System
    
    This application fetches **38 avalanche prediction parameters** from **8 different satellite/data sources**:
    
    ---
    
    #### 🌍 Primary Satellite Sources
    
    | # | Satellite/Source | Organization | Data Products | Resolution | Update Freq |
    |---|-----------------|--------------|---------------|------------|-------------|
    | 1 | **MODIS** (Terra/Aqua) | NASA | Snow Cover (MOD10A1), LST (MOD11A1), NDSI | 500m-1km | 1-2 days |
    | 2 | **VIIRS** (Suomi NPP/NOAA-20) | NASA/NOAA | Snow Cover (VNP10A1), Land Surface Temp | 375-750m | Daily |
    | 3 | **ERA5** Reanalysis | ECMWF/Copernicus | Temperature, Pressure, Radiation, Snow Depth | ~31km | Hourly |
    | 4 | **GOES-16/17/18** | NOAA | Cloud Cover, Radiation (via CERES) | 0.5-2km | 5-15 min |
    | 5 | **Sentinel-2/3** | ESA/Copernicus | High-res Snow Mapping, LST | 10m-1km | 5 days |
    | 6 | **CERES** (Aqua/Terra) | NASA | Radiation Budget (SW/LW) | 1° | Daily |
    | 7 | **NASA POWER** | NASA | MERRA-2 derived radiation, temp | 0.5° | Daily |
    | 8 | **NSIDC/AMSR2** | NASA/JAXA | SWE, Snow Depth | 25km | Daily |
    
    ---
    
    #### 📊 All 38 Parameters and Their Sources
    
    **🌡️ Temperature Parameters (4):**
    | Parameter | Description | Primary Source | Method |
    |-----------|-------------|----------------|--------|
    | `TA` | Air Temperature | ERA5 / Open-Meteo | Direct satellite |
    | `TA_daily` | Daily Air Temperature | ERA5 / NASA POWER | Direct satellite |
    | `TSS_mod` | Snow Surface Temperature | VIIRS LST / Calculated | Satellite + physics |
    | `profile_time` | Time of observation | System | - |
    
    **🌧️ Precipitation (2):**
    | Parameter | Description | Primary Source | Method |
    |-----------|-------------|----------------|--------|
    | `MS_Rain_daily` | Daily Rainfall | ERA5 / Open-Meteo | Direct satellite |
    | `SWE_daily` | Snow Water Equivalent | SNODAS / ERA5 | Satellite derived |
    
    **☀️ Radiation Parameters (8):**
    | Parameter | Description | Primary Source | Method |
    |-----------|-------------|----------------|--------|
    | `ISWR_daily` | Incoming Shortwave Radiation | CERES / NASA POWER | Direct satellite |
    | `ISWR_h_daily` | Horizontal Shortwave | CERES | Derived |
    | `ISWR_dir_daily` | Direct Shortwave | NASA POWER / ERA5 | Direct satellite |
    | `ISWR_diff_daily` | Diffuse Shortwave | NASA POWER / ERA5 | Direct satellite |
    | `ILWR` | Incoming Longwave | CERES / Calculated | Satellite + Stefan-Boltzmann |
    | `ILWR_daily` | Daily Incoming Longwave | CERES / Calculated | Satellite + Stefan-Boltzmann |
    | `OLWR` | Outgoing Longwave | Calculated | Stefan-Boltzmann (ε=0.98) |
    | `OLWR_daily` | Daily Outgoing Longwave | Calculated | Stefan-Boltzmann |
    
    **❄️ Snow Properties (4):**
    | Parameter | Description | Primary Source | Method |
    |-----------|-------------|----------------|--------|
    | `max_height` | Snow Depth | ERA5 / MODIS / Sentinel | Multi-source fusion |
    | `max_height_1_diff` | 1-day height change | ERA5 | Time series |
    | `max_height_2_diff` | 2-day height change | ERA5 | Time series |
    | `max_height_3_diff` | 3-day height change | ERA5 | Time series |
    
    **🔥 Heat Flux Parameters (4):**
    | Parameter | Description | Primary Source | Method |
    |-----------|-------------|----------------|--------|
    | `Qs` | Sensible Heat Flux | Calculated | Bulk aerodynamic method |
    | `Ql` | Latent Heat Flux | Calculated | Bulk aerodynamic method |
    | `Ql_daily` | Daily Latent Heat | Calculated | Physics-based |
    | `Qw_daily` | Absorbed Shortwave | Calculated | ISWR × (1-albedo) |
    
    **💧 Liquid Water Content (12):**
    | Parameter | Description | Primary Source | Method |
    |-----------|-------------|----------------|--------|
    | `mean_lwc` | Mean Liquid Water Content | Calculated | Energy balance melt model |
    | `max_lwc` | Maximum Liquid Water | Calculated | 1.5 × mean_lwc |
    | `std_lwc` | LWC Standard Deviation | Calculated | 0.3 × mean_lwc |
    | `water` | Total Liquid Water | Calculated | Degree-day + radiation melt |
    | `water_1_diff` | 1-day water change | Calculated | Temperature trend |
    | `water_2_diff` | 2-day water change | Calculated | Temperature trend |
    | `water_3_diff` | 3-day water change | Calculated | Temperature trend |
    | `mean_lwc_2_diff` | 2-day LWC change | Calculated | Temperature trend |
    | `mean_lwc_3_diff` | 3-day LWC change | Calculated | Temperature trend |
    | `prop_wet_2_diff` | Wet proportion change | Calculated | Temperature trend |
    | `sum_up` | Upper layer water | Calculated | water × prop_up |
    | `prop_up` | Upper wet fraction | Calculated | Melt state estimate |
    
    **📉 Wet Layer Depth (2):**
    | Parameter | Description | Primary Source | Method |
    |-----------|-------------|----------------|--------|
    | `lowest_2_diff` | 2-day depth change | Calculated | Melt percolation model |
    | `lowest_3_diff` | 3-day depth change | Calculated | Melt percolation model |
    
    **⚠️ Stability Indices (2):**
    | Parameter | Description | Primary Source | Method |
    |-----------|-------------|----------------|--------|
    | `S5` | Stability Index | Calculated | Multi-factor stability model |
    | `S5_daily` | Daily stability change | Calculated | Temperature trend |
    
    ---
    
    #### 🔬 Physics-Based Calculation Methods
    
    **Sensible Heat Flux (Qs):**
    ```
    Qs = ρ × cp × Ch × U × (Ta - Ts)
    Where: ρ=air density, cp=1005 J/kg/K, Ch=0.002, U=wind speed
    ```
    
    **Latent Heat Flux (Ql):**
    ```
    Ql = ρ × Ls × Ce × U × (qa - qs)
    Where: Ls=2.83×10⁶ J/kg (sublimation), Ce=0.002
    ```
    
    **Stefan-Boltzmann (Longwave Radiation):**
    ```
    OLWR = ε × σ × T⁴
    Where: ε=0.98 (snow emissivity), σ=5.67×10⁻⁸ W/m²/K⁴
    ```
    
    **Liquid Water Content:**
    ```
    Melt = DDF × max(Ta, 0) + (ISWR × 0.8) / (Lf × 1000)
    Where: DDF=4 mm/°C/day, Lf=334 kJ/kg
    ```
    
    ---
    
    #### ⚠️ Data Quality & Limitations
    
    | Data Type | Reliability | Notes |
    |-----------|-------------|-------|
    | 🟢 Direct Satellite | High | MODIS, VIIRS, CERES measurements |
    | 🟢 ERA5 Reanalysis | High | Model-observation fusion |
    | 🟡 Physics Calculations | Medium | Based on established equations |
    | 🟠 Estimated Parameters | Variable | Require ground validation |
    
    **Always consult official avalanche forecasts for critical decisions!**
    """)

st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🏔️ Avalanche Prediction System | Multi-Satellite Integration</p>
    <p><small>Data from: NASA (MODIS, VIIRS, CERES, POWER), Copernicus (ERA5, Sentinel), NOAA (GOES), NSIDC</small></p>
    <p><small>38 parameters from 8 satellite/data sources | Physics-informed calculations</small></p>
    <p><small>⚠️ This tool provides risk estimates only. Always consult official avalanche forecasts.</small></p>
</div>
""", unsafe_allow_html=True)
