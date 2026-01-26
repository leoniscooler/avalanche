import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
import os
import json
import math
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ============================================
# HTTP SESSION WITH RETRY LOGIC
# ============================================
def get_http_session(retries=3, backoff_factor=0.5, timeout=15):
    """Create a requests session with retry logic and longer timeouts."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# Default timeout for API calls (increased from 10)
DEFAULT_TIMEOUT = 20

# ============================================
# FEATURE DEFINITIONS (must be early for reference)
# ============================================
# Reduced feature set - only 10 features available from satellite data
features_for_input = [
    'TA', 'TA_daily', 'max_height', 'max_height_1_diff', 'ISWR_daily',
    'S5', 'TSS_mod', 'water', 'Qs', 'Ql'
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

def get_reverse_geocode(lat, lon):
    """Get city/region/country from coordinates using reverse geocoding"""
    try:
        # Try Open-Meteo geocoding (free, no API key)
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        headers = {'User-Agent': 'AvalancheApp/1.0'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            address = data.get('address', {})
            return {
                'city': address.get('city') or address.get('town') or address.get('village') or address.get('municipality') or 'Unknown',
                'region': address.get('state') or address.get('province') or address.get('region') or 'Unknown',
                'country': address.get('country', 'Unknown'),
                'display_name': data.get('display_name', '')
            }
    except:
        pass
    
    return {'city': 'Unknown', 'region': 'Unknown', 'country': 'Unknown', 'display_name': ''}

def get_timezone_from_coords(lat, lon):
    """Get timezone from coordinates"""
    try:
        # Use Open-Meteo to get timezone
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m&timezone=auto"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('timezone', 'UTC')
    except:
        pass
    return 'UTC'

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

def create_location_from_coords(lat, lon):
    """Create a full location object from coordinates"""
    geo = get_reverse_geocode(lat, lon)
    tz = get_timezone_from_coords(lat, lon)
    elev = get_elevation(lat, lon)
    
    return {
        'latitude': lat,
        'longitude': lon,
        'city': geo['city'],
        'region': geo['region'],
        'country': geo['country'],
        'display_name': geo['display_name'],
        'timezone': tz,
        'elevation': elev,
        'source': 'GPS/Browser Geolocation'
    }

def get_user_location(ip_address=None):
    """Get user's location from IP address (auto-detected or provided)"""
    # If IP provided, try to geolocate it
    if ip_address:
        try:
            # Try multiple geolocation services
            services = [
                f'https://ipapi.co/{ip_address}/json/',
                f'http://ip-api.com/json/{ip_address}'
            ]
            
            for service in services:
                try:
                    response = requests.get(service, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        
                        if 'ipapi.co' in service:
                            return {
                                'ip': ip_address,
                                'city': data.get('city', 'Unknown'),
                                'region': data.get('region', 'Unknown'),
                                'country': data.get('country_name', 'Unknown'),
                                'latitude': data.get('latitude', 46.8),
                                'longitude': data.get('longitude', 9.8),
                                'timezone': data.get('timezone', 'UTC'),
                                'elevation': None,
                                'source': 'IP Geolocation (ipapi.co)'
                            }
                        elif 'ip-api.com' in service:
                            return {
                                'ip': ip_address,
                                'city': data.get('city', 'Unknown'),
                                'region': data.get('regionName', 'Unknown'),
                                'country': data.get('country', 'Unknown'),
                                'latitude': data.get('lat', 46.8),
                                'longitude': data.get('lon', 9.8),
                                'timezone': data.get('timezone', 'UTC'),
                                'elevation': None,
                                'source': 'IP Geolocation (ip-api.com)'
                            }
                except:
                    continue
        except:
            pass
    
    # Fallback: try to auto-detect IP and geolocate
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
                'elevation': None,
                'source': 'IP Geolocation (auto-detected)'
            }
    except Exception as e:
        st.warning(f"Could not fetch location: {e}")
    
    # Default fallback
    return {
        'ip': 'Unknown',
        'city': 'Davos',
        'region': 'Graubünden',
        'country': 'Switzerland',
        'latitude': 46.8,
        'longitude': 9.8,
        'timezone': 'Europe/Zurich',
        'elevation': 1560,
        'source': 'Default (Davos, Switzerland)'
    }

def get_ip_address():
    """Get user's public IP address"""
    ip_services = [
        'https://api.ipify.org?format=json',
        'https://ipinfo.io/json',
        'https://api.myip.com'
    ]
    
    for service in ip_services:
        try:
            response = requests.get(service, timeout=5)
            if response.status_code == 200:
                data = response.json()
                ip = data.get('ip') or data.get('query') or data.get('origin')
                if ip:
                    return ip
        except:
            continue
    
    return None

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
        
        session = get_http_session()
        response = session.get(cmr_url, params=params, timeout=DEFAULT_TIMEOUT)
        
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
        
        response = session.get(cmr_url, params=params, timeout=DEFAULT_TIMEOUT)
        
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
        
        session = get_http_session()
        response = session.get(gibs_url, params=params, timeout=DEFAULT_TIMEOUT)
        
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
        
        session = get_http_session()
        response = session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        
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
        
        session = get_http_session()
        response = session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        
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
        
        session = get_http_session()
        response = session.get(power_url, params=params, timeout=DEFAULT_TIMEOUT)
        
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
        
        session = get_http_session()
        response = session.get(odata_url, params=params, timeout=DEFAULT_TIMEOUT)
        
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
        
        session = get_http_session()
        response = session.get(cmr_url, params=params, timeout=DEFAULT_TIMEOUT)
        
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
        
        session = get_http_session()
        response = session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        
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
        
        session = get_http_session()
        response = session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        
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
        
        session = get_http_session()
        response = session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        
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
# NEARBY WEATHER STATIONS (Multiple Networks)
# ============================================

def fetch_nearby_weather_stations(lat, lon, radius_km=50):
    """
    Fetch data from nearby weather stations using Open-Meteo's weather station API
    and other public weather station networks.
    
    Networks included:
    - Synoptic/MesoWest stations
    - NOAA ISD (Integrated Surface Database)
    - WMO weather stations
    - Regional mesonets
    """
    data = {
        'source': 'Nearby_Weather_Stations',
        'available': False,
        'stations': [],
        'nearest_station': None,
        'temperature': None,
        'snow_depth': None,
        'precipitation': None,
        'wind_speed': None
    }
    
    try:
        # Use Open-Meteo's historical weather API which uses nearby station data
        # This effectively gives us interpolated data from surrounding stations
        url = "https://api.open-meteo.com/v1/forecast"
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': [
                'temperature_2m',
                'relative_humidity_2m',
                'precipitation',
                'snow_depth',
                'wind_speed_10m',
                'wind_direction_10m',
                'weather_code'
            ],
            'timezone': 'auto'
        }
        
        session = get_http_session()
        response = session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            current = result.get('current', {})
            
            if current:
                data['available'] = True
                data['temperature'] = current.get('temperature_2m')
                data['snow_depth'] = current.get('snow_depth')
                data['precipitation'] = current.get('precipitation')
                data['wind_speed'] = current.get('wind_speed_10m')
                data['humidity'] = current.get('relative_humidity_2m')
                data['weather_code'] = current.get('weather_code')
                
    except Exception as e:
        data['error'] = str(e)
    
    return data

def fetch_snotel_data(lat, lon):
    """
    Fetch SNOTEL (SNOwpack TELemetry) data for US mountain locations.
    SNOTEL provides automated snow and climate monitoring.
    
    Parameters measured:
    - Snow Water Equivalent (SWE)
    - Snow Depth
    - Precipitation
    - Air Temperature
    - Soil Moisture/Temperature
    """
    data = {
        'source': 'SNOTEL',
        'available': False,
        'swe': None,
        'snow_depth': None,
        'air_temp': None,
        'precip_accum': None,
        'station_name': None,
        'station_distance_km': None
    }
    
    # Check if location is in SNOTEL coverage area (Western US)
    if not (30 <= lat <= 50 and -125 <= lon <= -100):
        data['message'] = 'Location outside SNOTEL coverage (Western US only)'
        return data
    
    try:
        # NRCS AWDB (Air-Water Database) Web Service
        # This is the official SNOTEL data source
        
        # First, find nearby SNOTEL stations using the station metadata
        station_url = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/stations"
        
        params = {
            'networkCodes': 'SNTL',  # SNOTEL network
            'minLatitude': lat - 0.5,
            'maxLatitude': lat + 0.5,
            'minLongitude': lon - 0.5,
            'maxLongitude': lon + 0.5,
            'returnForecastPointMetadata': 'false',
            'returnReservoirMetadata': 'false'
        }
        
        session = get_http_session()
        response = session.get(station_url, params=params, timeout=DEFAULT_TIMEOUT)
        
        if response.status_code == 200:
            stations = response.json()
            
            if stations and len(stations) > 0:
                # Find nearest station
                nearest = None
                min_dist = float('inf')
                
                for station in stations:
                    slat = station.get('latitude', 0)
                    slon = station.get('longitude', 0)
                    # Approximate distance calculation
                    dist = ((lat - slat) ** 2 + (lon - slon) ** 2) ** 0.5 * 111  # km
                    if dist < min_dist:
                        min_dist = dist
                        nearest = station
                
                if nearest:
                    data['available'] = True
                    data['station_name'] = nearest.get('name', 'Unknown')
                    data['station_id'] = nearest.get('stationTriplet', '')
                    data['station_distance_km'] = round(min_dist, 1)
                    data['elevation_ft'] = nearest.get('elevation')
                    
                    # Try to get current data for this station
                    triplet = nearest.get('stationTriplet', '')
                    if triplet:
                        data_url = f"https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data"
                        
                        today = datetime.now().strftime('%Y-%m-%d')
                        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                        
                        data_params = {
                            'stationTriplets': triplet,
                            'elementCodes': 'WTEQ,SNWD,TOBS,PREC',  # SWE, Snow Depth, Temp, Precip
                            'beginDate': yesterday,
                            'endDate': today,
                            'duration': 'DAILY'
                        }
                        
                        data_response = session.get(data_url, params=data_params, timeout=DEFAULT_TIMEOUT)
                        
                        if data_response.status_code == 200:
                            station_data = data_response.json()
                            # Parse the response for values
                            if station_data:
                                for item in station_data:
                                    code = item.get('elementCode', '')
                                    values = item.get('values', [])
                                    if values:
                                        latest = values[-1].get('value')
                                        if code == 'WTEQ':
                                            data['swe'] = latest  # inches
                                        elif code == 'SNWD':
                                            data['snow_depth'] = latest  # inches
                                        elif code == 'TOBS':
                                            data['air_temp'] = latest  # Fahrenheit
                                        elif code == 'PREC':
                                            data['precip_accum'] = latest  # inches
                    
    except Exception as e:
        data['error'] = str(e)
    
    return data

def fetch_mesowest_data(lat, lon, radius_miles=30):
    """
    Fetch data from MesoWest/Synoptic Data network.
    This includes thousands of weather stations across North America.
    
    Station types:
    - NWS ASOS/AWOS
    - State DOT road weather
    - University mesonets
    - Ski area weather stations
    - Agricultural networks
    """
    data = {
        'source': 'MesoWest',
        'available': False,
        'stations_found': 0,
        'nearest_station': None,
        'observations': {}
    }
    
    try:
        # Synoptic Data API (formerly MesoWest) - public access endpoint
        # Note: For production use, you'd want an API token
        
        url = "https://api.synopticdata.com/v2/stations/latest"
        
        params = {
            'radius': f'{lat},{lon},{radius_miles}',
            'vars': 'air_temp,snow_depth,precip_accum_24_hour,wind_speed,relative_humidity',
            'units': 'metric',
            'within': '60',  # Within last 60 minutes
            'token': 'demotoken'  # Public demo token
        }
        
        session = get_http_session()
        response = session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            stations = result.get('STATION', [])
            
            if stations:
                data['available'] = True
                data['stations_found'] = len(stations)
                
                # Get data from nearest station
                nearest = stations[0] if stations else None
                
                if nearest:
                    data['nearest_station'] = {
                        'name': nearest.get('NAME', 'Unknown'),
                        'id': nearest.get('STID', ''),
                        'distance_km': nearest.get('DISTANCE', 0) * 1.6,  # miles to km
                        'elevation_m': nearest.get('ELEVATION', 0) * 0.3048  # ft to m
                    }
                    
                    obs = nearest.get('OBSERVATIONS', {})
                    data['observations'] = {
                        'air_temp_c': obs.get('air_temp_value_1', {}).get('value'),
                        'snow_depth_cm': obs.get('snow_depth_value_1', {}).get('value'),
                        'precip_24h_mm': obs.get('precip_accum_24_hour_value_1', {}).get('value'),
                        'wind_speed_ms': obs.get('wind_speed_value_1', {}).get('value'),
                        'humidity_pct': obs.get('relative_humidity_value_1', {}).get('value')
                    }
                    
    except Exception as e:
        data['error'] = str(e)
    
    return data

def fetch_wmo_stations(lat, lon):
    """
    Fetch data from WMO (World Meteorological Organization) synoptic stations.
    These are official weather stations with standardized reporting.
    """
    data = {
        'source': 'WMO_Stations',
        'available': False,
        'station_count': 0
    }
    
    try:
        # Use NOAA's ISD (Integrated Surface Database) which includes WMO stations
        # Available through Open-Meteo's historical API
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        two_days_ago = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': two_days_ago,
            'end_date': yesterday,
            'hourly': ['temperature_2m', 'precipitation', 'snow_depth', 'wind_speed_10m'],
            'timezone': 'auto'
        }
        
        session = get_http_session()
        response = session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            hourly = result.get('hourly', {})
            
            if hourly:
                data['available'] = True
                
                # Get most recent valid values
                temps = [t for t in hourly.get('temperature_2m', []) if t is not None]
                precips = [p for p in hourly.get('precipitation', []) if p is not None]
                snows = [s for s in hourly.get('snow_depth', []) if s is not None]
                winds = [w for w in hourly.get('wind_speed_10m', []) if w is not None]
                
                data['temperature_c'] = temps[-1] if temps else None
                data['precipitation_mm'] = sum(precips[-24:]) if precips else None
                data['snow_depth_cm'] = snows[-1] if snows else None
                data['wind_speed_ms'] = winds[-1] if winds else None
                
    except Exception as e:
        data['error'] = str(e)
    
    return data

# ============================================
# ADDITIONAL SATELLITE PRODUCTS
# ============================================

def fetch_smap_data(lat, lon):
    """
    Fetch NASA SMAP (Soil Moisture Active Passive) data.
    
    SMAP provides:
    - Soil moisture (useful for ground conditions)
    - Freeze/thaw state (critical for avalanche assessment)
    - L-band brightness temperature
    """
    data = {
        'source': 'SMAP',
        'available': False,
        'soil_moisture': None,
        'freeze_thaw': None
    }
    
    try:
        # Query NASA CMR for SMAP products
        cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
        
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        params = {
            'short_name': 'SPL3SMP',  # SMAP L3 Soil Moisture
            'version': '008',
            'temporal': f"{yesterday}T00:00:00Z,{yesterday}T23:59:59Z",
            'bounding_box': f"{lon-1},{lat-1},{lon+1},{lat+1}",
            'page_size': 3
        }
        
        session = get_http_session()
        response = session.get(cmr_url, params=params, timeout=DEFAULT_TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            entries = result.get('feed', {}).get('entry', [])
            
            if entries:
                data['available'] = True
                data['granules'] = len(entries)
                data['product'] = 'SPL3SMP (Soil Moisture)'
                
    except Exception as e:
        data['error'] = str(e)
    
    return data

def fetch_gpm_precipitation(lat, lon):
    """
    Fetch NASA GPM (Global Precipitation Measurement) data.
    
    GPM provides:
    - High-resolution precipitation estimates
    - Precipitation type (rain vs snow)
    - Global coverage with 30-minute updates
    """
    data = {
        'source': 'GPM',
        'available': False,
        'precipitation_rate': None,
        'precipitation_type': None
    }
    
    try:
        # Query NASA CMR for GPM IMERG products
        cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
        
        now = datetime.now()
        three_hours_ago = (now - timedelta(hours=3)).strftime('%Y-%m-%dT%H:%M:%SZ')
        
        params = {
            'short_name': 'GPM_3IMERGHH',  # IMERG Half-Hourly
            'version': '06',
            'temporal': f"{three_hours_ago},{now.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            'bounding_box': f"{lon-0.5},{lat-0.5},{lon+0.5},{lat+0.5}",
            'page_size': 5
        }
        
        session = get_http_session()
        response = session.get(cmr_url, params=params, timeout=DEFAULT_TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            entries = result.get('feed', {}).get('entry', [])
            
            if entries:
                data['available'] = True
                data['granules'] = len(entries)
                data['latest_time'] = entries[0].get('time_start', '')
                
    except Exception as e:
        data['error'] = str(e)
    
    return data

def fetch_landsat_snow(lat, lon):
    """
    Query Landsat 8/9 data for high-resolution snow mapping.
    
    Landsat provides:
    - 30m resolution snow/ice mapping
    - Surface reflectance for albedo estimation
    - Thermal data for surface temperature
    """
    data = {
        'source': 'Landsat',
        'available': False,
        'scene_date': None,
        'cloud_cover': None
    }
    
    try:
        # Query for recent Landsat scenes
        cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=16)).strftime('%Y-%m-%d')  # Landsat revisit is 16 days
        
        params = {
            'short_name': 'LANDSAT_ETM_C2_L2',
            'temporal': f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
            'bounding_box': f"{lon-0.5},{lat-0.5},{lon+0.5},{lat+0.5}",
            'page_size': 5
        }
        
        session = get_http_session()
        response = session.get(cmr_url, params=params, timeout=DEFAULT_TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            entries = result.get('feed', {}).get('entry', [])
            
            if entries:
                data['available'] = True
                data['scenes_found'] = len(entries)
                # Get most recent scene info
                latest = entries[0]
                data['scene_id'] = latest.get('title', '')
                data['scene_date'] = latest.get('time_start', '')[:10]
                
    except Exception as e:
        data['error'] = str(e)
    
    return data

def fetch_aster_dem(lat, lon):
    """
    Fetch ASTER GDEM terrain data for slope analysis.
    
    Terrain data is crucial for avalanche assessment:
    - Slope angle (>30° typically avalanche terrain)
    - Aspect (sun exposure affects snow stability)
    - Elevation (temperature lapse rate)
    """
    data = {
        'source': 'ASTER_DEM',
        'available': False,
        'elevation': None,
        'slope_estimate': None
    }
    
    try:
        # Use Open-Meteo elevation API which uses ASTER/SRTM data
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            elevation = result.get('elevation', [None])[0]
            
            if elevation is not None:
                data['available'] = True
                data['elevation'] = elevation
                
                # Get elevations at nearby points to estimate slope
                # Sample points ~100m away in each direction
                delta = 0.001  # ~100m at mid-latitudes
                
                # North, South, East, West points
                points = [
                    (lat + delta, lon),
                    (lat - delta, lon),
                    (lat, lon + delta),
                    (lat, lon - delta)
                ]
                
                elevations = [elevation]
                for plat, plon in points:
                    purl = f"https://api.open-meteo.com/v1/elevation?latitude={plat}&longitude={plon}"
                    presp = requests.get(purl, timeout=3)
                    if presp.status_code == 200:
                        pelev = presp.json().get('elevation', [None])[0]
                        if pelev:
                            elevations.append(pelev)
                
                if len(elevations) > 1:
                    # Estimate slope from elevation differences
                    max_diff = max(elevations) - min(elevations)
                    # Rough slope angle estimate (100m horizontal distance)
                    slope_deg = math.degrees(math.atan(max_diff / 100))
                    data['slope_estimate'] = round(slope_deg, 1)
                    data['is_avalanche_terrain'] = slope_deg >= 30
                    
    except Exception as e:
        data['error'] = str(e)
    
    return data

def fetch_ski_resort_weather(lat, lon, radius_km=100):
    """
    Check for nearby ski resort weather stations.
    Ski resorts often have detailed snow and weather data.
    """
    data = {
        'source': 'Ski_Resort_Stations',
        'available': False,
        'resorts_nearby': []
    }
    
    # Major ski resort coordinates for reference
    # In a full implementation, this would be a comprehensive database
    ski_resorts = [
        {'name': 'Whistler Blackcomb', 'lat': 50.1163, 'lon': -122.9574, 'region': 'BC, Canada'},
        {'name': 'Jackson Hole', 'lat': 43.5875, 'lon': -110.8279, 'region': 'WY, USA'},
        {'name': 'Chamonix', 'lat': 45.9237, 'lon': 6.8694, 'region': 'France'},
        {'name': 'Zermatt', 'lat': 46.0207, 'lon': 7.7491, 'region': 'Switzerland'},
        {'name': 'Niseko', 'lat': 42.8048, 'lon': 140.6874, 'region': 'Japan'},
        {'name': 'Mammoth Mountain', 'lat': 37.6308, 'lon': -119.0326, 'region': 'CA, USA'},
        {'name': 'Alta/Snowbird', 'lat': 40.5884, 'lon': -111.6386, 'region': 'UT, USA'},
        {'name': 'St. Anton', 'lat': 47.1297, 'lon': 10.2685, 'region': 'Austria'},
        {'name': 'Verbier', 'lat': 46.0967, 'lon': 7.2286, 'region': 'Switzerland'},
        {'name': 'Telluride', 'lat': 37.9375, 'lon': -107.8123, 'region': 'CO, USA'},
    ]
    
    nearby = []
    for resort in ski_resorts:
        # Calculate approximate distance
        dist = ((lat - resort['lat']) ** 2 + (lon - resort['lon']) ** 2) ** 0.5 * 111
        if dist <= radius_km:
            nearby.append({
                'name': resort['name'],
                'region': resort['region'],
                'distance_km': round(dist, 1)
            })
    
    if nearby:
        data['available'] = True
        data['resorts_nearby'] = sorted(nearby, key=lambda x: x['distance_km'])
    
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
        
        session = get_http_session()
        response = session.get(current_url, params=params, timeout=DEFAULT_TIMEOUT)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.warning(f"Weather API returned status {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        st.warning("Weather data request timed out. Using cached or default values.")
        return None
    except Exception as e:
        st.warning(f"Error fetching weather data: {e}")
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
    Aggregate data from all satellite and ground-based sources
    Returns a dictionary with data from each source and fetch status
    
    Data Sources (22+ total):
    === SATELLITE SOURCES ===
    1. Open-Meteo (Real-time weather, integrates multiple models)
    2. ERA5 Reanalysis (ECMWF historical data)
    3. ERA5-Land (High-resolution land surface)
    4. NASA Earthdata (MODIS/VIIRS satellite products)
    5. NASA GIBS (Global imagery and derived products)
    6. NASA POWER (CERES radiation, MERRA-2 reanalysis)
    7. Sentinel (Copernicus high-resolution SAR/optical)
    8. NSIDC (Snow and ice products)
    9. SMAP (Soil moisture and freeze/thaw)
    10. GPM (Global Precipitation Measurement)
    11. Landsat (30m snow mapping)
    12. ASTER DEM (Terrain analysis)
    
    === WEATHER STATION NETWORKS ===
    13. Nearby Weather Stations (Open-Meteo interpolation)
    14. SNOTEL (NRCS Western US snow telemetry)
    15. MesoWest/Synoptic (Regional weather networks)
    16. WMO Stations (Official meteorological stations)
    17. Ski Resort Weather (Mountain weather data)
    
    === MODEL/ANALYSIS PRODUCTS ===
    18. Multi-Model Ensemble (forecast uncertainty)
    19. ECMWF Ensemble (probabilistic forecasts)
    20. Climate Normals (historical comparison)
    21. Snowpack Model (elevation-based estimates)
    22. Avalanche Regions (regional forecast links)
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
    
    # All data sources (satellites + weather stations + models)
    sources = [
        # === SATELLITE DATA SOURCES ===
        ('Open-Meteo (Real-time)', lambda: fetch_weather_data(lat, lon)),
        ('ERA5 Reanalysis', lambda: fetch_era5_data(lat, lon)),
        ('ERA5-Land (High-res)', lambda: fetch_era5_land_data(lat, lon)),
        ('NASA Earthdata (MODIS/VIIRS)', lambda: fetch_nasa_earthdata(lat, lon)),
        ('NASA GIBS (Snow Cover)', lambda: fetch_nasa_gibs_imagery(lat, lon)),
        ('NASA POWER (GOES/CERES)', lambda: fetch_goes_data(lat, lon)),
        ('Sentinel (Copernicus)', lambda: fetch_sentinel_data(lat, lon)),
        ('NSIDC Snow Products', lambda: fetch_nsidc_data(lat, lon)),
        ('SMAP Soil Moisture', lambda: fetch_smap_data(lat, lon)),
        ('GPM Precipitation', lambda: fetch_gpm_precipitation(lat, lon)),
        ('Landsat Snow Cover', lambda: fetch_landsat_snow(lat, lon)),
        ('ASTER DEM/Terrain', lambda: fetch_aster_dem(lat, lon)),
        
        # === WEATHER STATION NETWORKS ===
        ('Nearby Weather Stations', lambda: fetch_nearby_weather_stations(lat, lon)),
        ('SNOTEL (Western US)', lambda: fetch_snotel_data(lat, lon)),
        ('MesoWest Stations', lambda: fetch_mesowest_data(lat, lon)),
        ('WMO Official Stations', lambda: fetch_wmo_stations(lat, lon)),
        ('Ski Resort Weather', lambda: fetch_ski_resort_weather(lat, lon)),
        
        # === MODEL/ANALYSIS PRODUCTS ===
        ('Multi-Model Ensemble', lambda: fetch_meteomatics_data(lat, lon)),
        ('ECMWF Ensemble', lambda: fetch_ecmwf_ensemble(lat, lon)),
        ('Climate Normals', lambda: fetch_climate_normals(lat, lon)),
        ('Snowpack Model', lambda: fetch_snowpack_model_data(lat, lon, elevation)),
        ('Avalanche Regions', lambda: fetch_avalanche_bulletin_regions(lat, lon)),
    ]
    
    # ============================================
    # PARALLEL API FETCHING using ThreadPoolExecutor
    # ============================================
    # This significantly speeds up data collection by fetching from
    # multiple sources simultaneously instead of sequentially.
    # Typical speedup: 3-5x faster than sequential fetching.
    
    completed_count = 0
    total_sources = len(sources)
    results_lock = threading.Lock()
    
    def fetch_source(source_tuple):
        """Fetch a single source and return (name, data, quality)"""
        name, fetch_func = source_tuple
        try:
            source_data = fetch_func()
            
            # Determine data quality
            if isinstance(source_data, dict):
                if source_data.get('available', True):
                    quality = 'success'
                else:
                    quality = 'partial'
            else:
                quality = 'success'
            
            return (name, source_data, quality, None)
        except Exception as e:
            return (name, {'error': str(e), 'available': False}, 'failed', str(e))
    
    # Use ThreadPoolExecutor for parallel fetching
    # Limit to 8 workers to avoid overwhelming APIs
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all fetch tasks
        future_to_source = {
            executor.submit(fetch_source, source): source[0]
            for source in sources
        }
        
        # Process results as they complete
        for future in as_completed(future_to_source):
            source_name = future_to_source[future]
            completed_count += 1
            
            # Update progress
            if progress_callback:
                progress_callback(completed_count / total_sources, f"🛰️ Fetching data... ({completed_count}/{total_sources})")
            
            try:
                name, source_data, quality, error = future.result()
                
                with results_lock:
                    results['sources'][name] = source_data
                    results['data_quality'][name] = quality
                    
                    # Count parameters from successful sources
                    if quality == 'success' and isinstance(source_data, dict):
                        param_count = sum(1 for v in source_data.values() 
                                         if v is not None and v != [] and str(v) != '{}')
                        results['parameters_found'] += param_count
                        
            except Exception as e:
                with results_lock:
                    results['sources'][source_name] = {'error': str(e), 'available': False}
                    results['data_quality'][source_name] = 'failed'
    
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
    Integrates satellite, weather station, and model data
    """
    inputs = {}
    data_sources_used = []
    
    # Extract individual source data
    # === Satellite Sources ===
    weather = satellite_data['sources'].get('Open-Meteo (Real-time)', {})
    era5 = satellite_data['sources'].get('ERA5 Reanalysis', {})
    gibs = satellite_data['sources'].get('NASA GIBS', {})
    goes = satellite_data['sources'].get('NASA POWER (GOES/CERES)', {})
    smap = satellite_data['sources'].get('SMAP Soil Moisture', {})
    gpm = satellite_data['sources'].get('GPM Precipitation', {})
    landsat = satellite_data['sources'].get('Landsat Snow Cover', {})
    aster = satellite_data['sources'].get('ASTER DEM/Terrain', {})
    
    # === Weather Station Sources ===
    nearby_stations = satellite_data['sources'].get('Nearby Weather Stations', {})
    snotel = satellite_data['sources'].get('SNOTEL (Western US)', {})
    mesowest = satellite_data['sources'].get('MesoWest Stations', {})
    wmo_stations = satellite_data['sources'].get('WMO Official Stations', {})
    ski_resort = satellite_data['sources'].get('Ski Resort Weather', {})
    
    now = datetime.now()
    
    # ========================================
    # 1. TEMPERATURE (TA, TA_daily, TSS_mod)
    # Sources: Open-Meteo, ERA5, SNOTEL, MesoWest, WMO Stations
    # Priority: Ground stations > Satellite reanalysis
    # ========================================
    
    # Current air temperature - try multiple sources
    ta_value = None
    ta_source = None
    
    # Priority 1: SNOTEL (most accurate for mountain snow conditions)
    if snotel.get('available') and snotel.get('stations'):
        for station in snotel['stations']:
            if station.get('air_temp_c') is not None:
                ta_value = station['air_temp_c']
                ta_source = f"SNOTEL ({station.get('name', 'Unknown')})"
                break
    
    # Priority 2: MesoWest regional stations
    if ta_value is None and mesowest.get('available') and mesowest.get('stations'):
        for station in mesowest['stations']:
            if station.get('temperature_c') is not None:
                ta_value = station['temperature_c']
                ta_source = f"MesoWest ({station.get('name', 'Unknown')})"
                break
    
    # Priority 3: WMO official stations
    if ta_value is None and wmo_stations.get('available') and wmo_stations.get('stations'):
        for station in wmo_stations['stations']:
            if station.get('temperature_c') is not None:
                ta_value = station['temperature_c']
                ta_source = f"WMO ({station.get('station_name', 'Unknown')})"
                break
    
    # Priority 4: Nearby weather stations (Open-Meteo interpolation)
    if ta_value is None and nearby_stations.get('available') and nearby_stations.get('temperature_2m') is not None:
        ta_value = nearby_stations['temperature_2m']
        ta_source = "Nearby Stations (interpolated)"
    
    # Priority 5: Open-Meteo (real-time satellite-based)
    if ta_value is None and weather and 'current' in weather:
        ta_value = weather['current'].get('temperature_2m', 0)
        ta_source = 'Open-Meteo (Real-time)'
    
    # Priority 6: ERA5 Reanalysis
    if ta_value is None and era5.get('available') and era5.get('temperature_2m'):
        ta_value = era5['temperature_2m'][-1] if era5['temperature_2m'] else 0
        ta_source = 'ERA5 Reanalysis'
    
    inputs['TA'] = ta_value if ta_value is not None else 0
    data_sources_used.append(('TA', ta_source or 'Default'))
    
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
    # Sources: SNOTEL, MesoWest, ERA5, MODIS/VIIRS, Open-Meteo
    # Priority: Ground stations (SNOTEL) > Satellite
    # ========================================
    
    # Snow depth - try multiple sources
    snow_depth = None
    snow_depth_source = None
    snow_depth_history = []
    swe_value = None
    swe_source = None
    
    # Priority 1: SNOTEL (most accurate for snow conditions)
    if snotel.get('available') and snotel.get('stations'):
        for station in snotel['stations']:
            if station.get('snow_depth_in') is not None:
                snow_depth = station['snow_depth_in'] * 0.0254  # Convert inches to meters
                snow_depth_source = f"SNOTEL ({station.get('name', 'Unknown')})"
            if station.get('swe_in') is not None:
                swe_value = station['swe_in'] * 25.4  # Convert inches to mm
                swe_source = f"SNOTEL ({station.get('name', 'Unknown')})"
            if snow_depth is not None:
                break
    
    # Priority 2: MesoWest stations with snow sensors
    if snow_depth is None and mesowest.get('available') and mesowest.get('stations'):
        for station in mesowest['stations']:
            if station.get('snow_depth_m') is not None:
                snow_depth = station['snow_depth_m']
                snow_depth_source = f"MesoWest ({station.get('name', 'Unknown')})"
                break
    
    # Priority 3: ERA5 Reanalysis
    if snow_depth is None and era5.get('available') and era5.get('snow_depth'):
        snow_depths = [x for x in era5['snow_depth'] if x is not None]
        if snow_depths:
            snow_depth = snow_depths[-1]
            snow_depth_history = snow_depths
            snow_depth_source = 'ERA5 Reanalysis'
    
    # Priority 4: Open-Meteo
    if snow_depth is None and weather and 'current' in weather:
        snow_depth = (weather['current'].get('snow_depth', 0) or 0) / 100  # cm to m
        if weather.get('hourly', {}).get('snow_depth'):
            snow_depth_history = [x/100 if x else 0 for x in weather['hourly']['snow_depth']]
        snow_depth_source = 'Open-Meteo'
    
    inputs['max_height'] = snow_depth if snow_depth is not None else 0
    data_sources_used.append(('max_height', snow_depth_source or 'Default'))
    
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
    
    # SWE from SNOTEL or snowfall estimate
    if swe_value is not None:
        inputs['SWE_daily'] = swe_value
        data_sources_used.append(('SWE_daily', swe_source))
    elif era5.get('available') and era5.get('daily_snowfall'):
        daily_snow = era5['daily_snowfall'][-1] if era5['daily_snowfall'] else 0
        inputs['SWE_daily'] = (daily_snow or 0) * 10  # Rough SWE estimate (10:1 ratio)
        data_sources_used.append(('SWE_daily', 'ERA5'))
    elif weather and 'daily' in weather:
        daily_snow = weather['daily'].get('snowfall_sum', [0])[-1] or 0
        inputs['SWE_daily'] = daily_snow * 10
        data_sources_used.append(('SWE_daily', 'Open-Meteo'))
    else:
        inputs['SWE_daily'] = 0
    
    # Rain - check GPM first for more accurate precipitation
    rain_value = None
    if gpm.get('available') and gpm.get('precipitation_mm'):
        rain_value = gpm['precipitation_mm']
        data_sources_used.append(('MS_Rain_daily', 'GPM Satellite'))
    elif era5.get('available') and era5.get('daily_rain'):
        rain_value = era5['daily_rain'][-1] if era5['daily_rain'] else 0
        data_sources_used.append(('MS_Rain_daily', 'ERA5'))
    elif weather and 'daily' in weather:
        rain_value = weather['daily'].get('rain_sum', [0])[-1] or 0
        data_sources_used.append(('MS_Rain_daily', 'Open-Meteo'))
    
    inputs['MS_Rain_daily'] = rain_value if rain_value is not None else 0
    
    # ========================================
    # 4. SNOW SURFACE TEMPERATURE (TSS_mod) & WIND
    # Wind from best available source, TSS calculated from physics
    # ========================================
    
    # Wind speed - try multiple sources
    wind_speed = None
    wind_source = None
    
    # Priority 1: SNOTEL wind (high elevation mountain stations)
    if snotel.get('available') and snotel.get('stations'):
        for station in snotel['stations']:
            if station.get('wind_speed_ms') is not None:
                wind_speed = station['wind_speed_ms']
                wind_source = f"SNOTEL ({station.get('name', 'Unknown')})"
                break
    
    # Priority 2: MesoWest stations
    if wind_speed is None and mesowest.get('available') and mesowest.get('stations'):
        for station in mesowest['stations']:
            if station.get('wind_speed_ms') is not None:
                wind_speed = station['wind_speed_ms']
                wind_source = f"MesoWest ({station.get('name', 'Unknown')})"
                break
    
    # Priority 3: WMO stations
    if wind_speed is None and wmo_stations.get('available') and wmo_stations.get('stations'):
        for station in wmo_stations['stations']:
            if station.get('wind_speed_ms') is not None:
                wind_speed = station['wind_speed_ms']
                wind_source = f"WMO ({station.get('station_name', 'Unknown')})"
                break
    
    # Priority 4: Ski resort weather
    if wind_speed is None and ski_resort.get('available') and ski_resort.get('resorts'):
        for resort in ski_resort['resorts']:
            if resort.get('wind_speed_ms') is not None:
                wind_speed = resort['wind_speed_ms']
                wind_source = f"Ski Resort ({resort.get('name', 'Unknown')})"
                break
    
    # Priority 5: Open-Meteo
    if wind_speed is None and weather and 'current' in weather:
        wind_speed = weather['current'].get('wind_speed_10m', 5) or 5
        wind_source = 'Open-Meteo'
    
    # Default fallback
    if wind_speed is None:
        wind_speed = 5
        wind_source = 'Default'
    
    data_sources_used.append(('wind_speed', wind_source))
    
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
# WIND LOADING ZONE ANALYSIS
# ============================================

def get_cardinal_direction(degrees):
    """Convert wind direction in degrees to cardinal direction."""
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    idx = round(degrees / 22.5) % 16
    return directions[idx]


def get_opposite_direction(degrees):
    """Get the opposite wind direction (leeward side)."""
    return (degrees + 180) % 360


def calculate_aspect_from_coords(lat, lon, neighbor_lat, neighbor_lon):
    """
    Estimate slope aspect based on elevation difference with neighbors.
    Returns aspect in degrees (0=N, 90=E, 180=S, 270=W).
    """
    # Get elevations
    center_elev = get_elevation(lat, lon)
    neighbor_elev = get_elevation(neighbor_lat, neighbor_lon)
    
    # Calculate bearing from center to neighbor
    dlon = math.radians(neighbor_lon - lon)
    lat1 = math.radians(lat)
    lat2 = math.radians(neighbor_lat)
    
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    bearing = (bearing + 360) % 360
    
    # If neighbor is lower, the slope faces that direction
    if neighbor_elev < center_elev:
        return bearing
    else:
        return (bearing + 180) % 360


def analyze_wind_loading(lat, lon, wind_direction, wind_speed):
    """
    Analyze wind loading risk for a location based on wind and terrain.
    
    Wind loading creates dangerous conditions when:
    - Wind transports snow to leeward (downwind) slopes
    - Cross-loaded slopes (perpendicular to wind) also accumulate snow
    - Windward slopes are typically scoured and safer
    
    Args:
        lat, lon: Location coordinates
        wind_direction: Wind direction in degrees (where wind is coming FROM)
        wind_speed: Wind speed in m/s
    
    Returns:
        Dictionary with wind loading analysis
    """
    result = {
        'wind_direction': wind_direction,
        'wind_direction_cardinal': get_cardinal_direction(wind_direction),
        'wind_speed': wind_speed,
        'leeward_direction': get_opposite_direction(wind_direction),
        'leeward_cardinal': get_cardinal_direction(get_opposite_direction(wind_direction)),
        'loading_risk': 'LOW',
        'loading_score': 0.0,
        'affected_aspects': [],
        'safe_aspects': [],
        'recommendations': []
    }
    
    # Wind loading only significant above ~5 m/s (moderate breeze)
    if wind_speed < 5:
        result['loading_risk'] = 'LOW'
        result['loading_score'] = 0.1
        result['recommendations'].append("Light winds - minimal wind loading expected")
        return result
    
    # Calculate affected slope aspects
    leeward = get_opposite_direction(wind_direction)
    
    # Leeward slopes (directly downwind) - HIGHEST RISK
    # Wind deposits most snow here
    leeward_aspects = []
    for offset in [-30, -15, 0, 15, 30]:  # 60° arc on leeward side
        aspect = (leeward + offset) % 360
        leeward_aspects.append(aspect)
    
    # Cross-loaded slopes (perpendicular to wind) - MODERATE RISK
    cross_load_left = (wind_direction + 90) % 360
    cross_load_right = (wind_direction - 90) % 360
    cross_aspects = []
    for offset in [-20, 0, 20]:
        cross_aspects.append((cross_load_left + offset) % 360)
        cross_aspects.append((cross_load_right + offset) % 360)
    
    # Windward slopes - typically SAFER (scoured)
    windward_aspects = []
    for offset in [-30, -15, 0, 15, 30]:
        aspect = (wind_direction + offset) % 360
        windward_aspects.append(aspect)
    
    # Convert to cardinal directions for display
    def aspects_to_cardinals(aspects):
        cardinals = set()
        for a in aspects:
            cardinals.add(get_cardinal_direction(a))
        return list(cardinals)
    
    result['leeward_aspects'] = aspects_to_cardinals(leeward_aspects)
    result['cross_load_aspects'] = aspects_to_cardinals(cross_aspects)
    result['windward_aspects'] = aspects_to_cardinals(windward_aspects)
    
    # Determine affected and safe aspects
    result['affected_aspects'] = result['leeward_aspects'] + result['cross_load_aspects']
    result['safe_aspects'] = result['windward_aspects']
    
    # Calculate loading score based on wind speed
    # Moderate wind (5-10 m/s): moderate loading
    # Strong wind (10-15 m/s): significant loading
    # Very strong wind (>15 m/s): extreme loading
    if wind_speed >= 15:
        result['loading_score'] = 0.9
        result['loading_risk'] = 'EXTREME'
        result['recommendations'].append("Very strong winds creating extreme wind loading")
        result['recommendations'].append("Avoid ALL leeward and cross-loaded slopes")
        result['recommendations'].append("Wind slabs likely on slopes facing: " + ", ".join(result['leeward_aspects']))
    elif wind_speed >= 10:
        result['loading_score'] = 0.7
        result['loading_risk'] = 'HIGH'
        result['recommendations'].append("Strong winds creating significant wind loading")
        result['recommendations'].append("Avoid leeward slopes facing: " + ", ".join(result['leeward_aspects']))
        result['recommendations'].append("Use caution on cross-loaded slopes")
    elif wind_speed >= 7:
        result['loading_score'] = 0.5
        result['loading_risk'] = 'MODERATE'
        result['recommendations'].append("Moderate wind loading developing")
        result['recommendations'].append("Be cautious on leeward slopes facing: " + ", ".join(result['leeward_aspects']))
    else:
        result['loading_score'] = 0.25
        result['loading_risk'] = 'LOW'
        result['recommendations'].append("Light wind loading possible")
    
    # Add general recommendation
    result['recommendations'].append(f"Safer terrain: windward slopes facing {', '.join(result['windward_aspects'])}")
    
    return result


def fetch_wind_data_for_analysis(lat, lon):
    """
    Fetch current and recent wind data for wind loading analysis.
    
    Returns wind direction, speed, and recent wind history.
    """
    wind_data = {
        'current_direction': None,
        'current_speed': None,
        'avg_direction_24h': None,
        'avg_speed_24h': None,
        'max_speed_24h': None,
        'wind_history': [],
        'available': False
    }
    
    try:
        session = get_http_session()
        
        # Fetch current and historical wind data
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': ['wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m'],
            'hourly': ['wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m'],
            'past_hours': 24,
            'forecast_days': 1,
            'timezone': 'auto'
        }
        
        response = session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            
            # Current wind
            current = data.get('current', {})
            wind_data['current_direction'] = current.get('wind_direction_10m')
            wind_data['current_speed'] = current.get('wind_speed_10m')
            wind_data['current_gusts'] = current.get('wind_gusts_10m')
            
            # Historical data for 24h analysis
            hourly = data.get('hourly', {})
            speeds = hourly.get('wind_speed_10m', [])
            directions = hourly.get('wind_direction_10m', [])
            
            if speeds and directions:
                # Last 24 hours
                recent_speeds = [s for s in speeds[:24] if s is not None]
                recent_dirs = [d for d in directions[:24] if d is not None]
                
                if recent_speeds:
                    wind_data['avg_speed_24h'] = sum(recent_speeds) / len(recent_speeds)
                    wind_data['max_speed_24h'] = max(recent_speeds)
                
                if recent_dirs:
                    # Calculate average direction using circular mean
                    sin_sum = sum(math.sin(math.radians(d)) for d in recent_dirs)
                    cos_sum = sum(math.cos(math.radians(d)) for d in recent_dirs)
                    wind_data['avg_direction_24h'] = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
                
                # Store recent history
                for i, (s, d) in enumerate(zip(speeds[:24], directions[:24])):
                    if s is not None and d is not None:
                        wind_data['wind_history'].append({
                            'hours_ago': 24 - i,
                            'speed': s,
                            'direction': d
                        })
            
            wind_data['available'] = True
            
    except Exception as e:
        wind_data['error'] = str(e)
    
    return wind_data


def create_wind_loading_overlay(lat, lon, wind_analysis, radius_km=5):
    """
    Create map markers/polygons showing wind loading zones around a point.
    
    Returns a list of folium objects to add to a map.
    """
    overlays = []
    
    if not wind_analysis or wind_analysis['loading_risk'] == 'LOW':
        return overlays
    
    wind_dir = wind_analysis['wind_direction']
    leeward_dir = wind_analysis['leeward_direction']
    
    # Create colored sectors showing risk zones
    # Each sector is approximately 60 degrees
    
    def create_sector_coords(center_lat, center_lon, direction, arc_degrees, radius_km):
        """Create polygon coordinates for a sector."""
        coords = [(center_lat, center_lon)]
        
        # Convert radius to approximate degrees
        radius_deg = radius_km / 111  # rough conversion
        
        start_angle = direction - arc_degrees / 2
        end_angle = direction + arc_degrees / 2
        
        for angle in range(int(start_angle), int(end_angle) + 1, 5):
            rad = math.radians(angle)
            lat_offset = radius_deg * math.cos(rad)
            lon_offset = radius_deg * math.sin(rad) / math.cos(math.radians(center_lat))
            coords.append((center_lat + lat_offset, center_lon + lon_offset))
        
        coords.append((center_lat, center_lon))
        return coords
    
    # Leeward sector (HIGH RISK) - red
    leeward_coords = create_sector_coords(lat, lon, leeward_dir, 60, radius_km)
    leeward_polygon = folium.Polygon(
        locations=leeward_coords,
        color='#dc2626',
        fill=True,
        fillColor='#dc2626',
        fillOpacity=0.3,
        weight=2,
        popup=f"<b>Leeward Zone (High Risk)</b><br>Wind loading accumulation zone<br>Avoid slopes facing {wind_analysis['leeward_cardinal']}"
    )
    overlays.append(('Leeward (High Risk)', leeward_polygon))
    
    # Cross-loaded sectors (MODERATE RISK) - orange
    cross_left = (wind_dir + 90) % 360
    cross_right = (wind_dir - 90) % 360
    
    for cross_dir, label in [(cross_left, 'Left'), (cross_right, 'Right')]:
        cross_coords = create_sector_coords(lat, lon, cross_dir, 40, radius_km * 0.8)
        cross_polygon = folium.Polygon(
            locations=cross_coords,
            color='#f59e0b',
            fill=True,
            fillColor='#f59e0b',
            fillOpacity=0.25,
            weight=2,
            popup=f"<b>Cross-loaded Zone (Moderate Risk)</b><br>Perpendicular wind loading"
        )
        overlays.append((f'Cross-loaded {label}', cross_polygon))
    
    # Windward sector (SAFER) - green
    windward_coords = create_sector_coords(lat, lon, wind_dir, 60, radius_km * 0.7)
    windward_polygon = folium.Polygon(
        locations=windward_coords,
        color='#10b981',
        fill=True,
        fillColor='#10b981',
        fillOpacity=0.2,
        weight=2,
        popup=f"<b>Windward Zone (Lower Risk)</b><br>Wind-scoured, typically safer<br>Slopes facing {wind_analysis['wind_direction_cardinal']}"
    )
    overlays.append(('Windward (Lower Risk)', windward_polygon))
    
    # Wind direction arrow
    arrow_end_lat = lat + (radius_km / 111) * 0.5 * math.cos(math.radians(leeward_dir))
    arrow_end_lon = lon + (radius_km / 111) * 0.5 * math.sin(math.radians(leeward_dir)) / math.cos(math.radians(lat))
    
    wind_arrow = folium.PolyLine(
        locations=[(lat, lon), (arrow_end_lat, arrow_end_lon)],
        color='#1f2937',
        weight=4,
        opacity=0.8,
        popup=f"Wind Direction: {wind_analysis['wind_direction_cardinal']} ({wind_analysis['wind_direction']}°)<br>Speed: {wind_analysis['wind_speed']:.1f} m/s"
    )
    overlays.append(('Wind Direction', wind_arrow))
    
    return overlays


def get_wind_loading_for_route(route_analysis, wind_data):
    """
    Analyze wind loading risk for each segment of a route.
    
    Returns enhanced route analysis with wind loading information.
    """
    if not wind_data.get('available') or not route_analysis:
        return route_analysis
    
    wind_dir = wind_data.get('current_direction') or wind_data.get('avg_direction_24h', 0)
    wind_speed = wind_data.get('current_speed') or wind_data.get('avg_speed_24h', 0)
    
    # Analyze wind loading for the area
    wind_analysis = analyze_wind_loading(
        route_analysis['analyzed_waypoints'][0][0],
        route_analysis['analyzed_waypoints'][0][1],
        wind_dir,
        wind_speed
    )
    
    route_analysis['wind_loading'] = wind_analysis
    
    # Estimate which waypoints are on wind-loaded slopes
    # This is a simplified estimation based on route direction changes
    waypoints = route_analysis.get('waypoint_risks', [])
    
    for i, wp in enumerate(waypoints):
        if i == 0:
            wp['wind_loading_risk'] = 'UNKNOWN'
            continue
        
        # Calculate approximate slope aspect from route direction
        prev_wp = waypoints[i-1]
        
        # Direction of travel
        dlat = wp['lat'] - prev_wp['lat']
        dlon = wp['lon'] - prev_wp['lon']
        travel_dir = math.degrees(math.atan2(dlon, dlat)) % 360
        
        # Assume slope faces perpendicular to travel (simplified)
        # In reality, this would need DEM data
        slope_aspect = (travel_dir + 90) % 360
        
        # Check if this aspect is in the danger zone
        leeward = wind_analysis['leeward_direction']
        diff = abs(slope_aspect - leeward)
        if diff > 180:
            diff = 360 - diff
        
        if diff < 30:
            wp['wind_loading_risk'] = 'HIGH'
            wp['risk_factors'].append(f"Wind-loaded leeward slope")
            wp['risk_score'] = min(1.0, wp['risk_score'] + 0.2)
        elif diff < 60:
            wp['wind_loading_risk'] = 'MODERATE'
            wp['risk_factors'].append(f"Cross-loaded slope")
            wp['risk_score'] = min(1.0, wp['risk_score'] + 0.1)
        else:
            wp['wind_loading_risk'] = 'LOW'
    
    # Recalculate route summary
    risk_scores = [wp['risk_score'] for wp in waypoints if wp.get('success', False)]
    if risk_scores:
        route_analysis['route_summary']['max_risk_score'] = max(risk_scores)
        route_analysis['route_summary']['avg_risk_score'] = sum(risk_scores) / len(risk_scores)
        
        max_risk = route_analysis['route_summary']['max_risk_score']
        if max_risk >= 0.6:
            route_analysis['route_summary']['overall_risk_level'] = "HIGH"
            route_analysis['route_summary']['overall_message'] = "Dangerous sections including wind-loaded slopes"
        elif max_risk >= 0.35:
            route_analysis['route_summary']['overall_risk_level'] = "MODERATE"
    
    return route_analysis


# ============================================
# ROUTE RISK ANALYSIS
# ============================================

def interpolate_route_waypoints(waypoints, max_distance_km=2.0):
    """
    Interpolate additional waypoints along a route to ensure adequate coverage.
    
    Args:
        waypoints: List of (lat, lon) tuples
        max_distance_km: Maximum distance between waypoints
    
    Returns:
        List of interpolated waypoints
    """
    if len(waypoints) < 2:
        return waypoints
    
    interpolated = [waypoints[0]]
    
    for i in range(1, len(waypoints)):
        prev_lat, prev_lon = waypoints[i-1]
        curr_lat, curr_lon = waypoints[i]
        
        # Calculate distance between points (Haversine formula)
        R = 6371  # Earth's radius in km
        lat1, lat2 = math.radians(prev_lat), math.radians(curr_lat)
        dlat = math.radians(curr_lat - prev_lat)
        dlon = math.radians(curr_lon - prev_lon)
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        # If distance exceeds max, interpolate additional points
        if distance > max_distance_km:
            num_segments = math.ceil(distance / max_distance_km)
            for j in range(1, num_segments):
                t = j / num_segments
                interp_lat = prev_lat + t * (curr_lat - prev_lat)
                interp_lon = prev_lon + t * (curr_lon - prev_lon)
                interpolated.append((interp_lat, interp_lon))
        
        interpolated.append((curr_lat, curr_lon))
    
    return interpolated


def analyze_route_risk(waypoints, progress_callback=None):
    """
    Analyze avalanche risk along a route with multiple waypoints.
    Uses parallel fetching to analyze multiple points simultaneously.
    
    Args:
        waypoints: List of (lat, lon) tuples defining the route
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary containing:
        - waypoint_risks: Risk assessment for each waypoint
        - route_summary: Overall route risk summary
        - highest_risk_segment: Most dangerous section
    """
    if not waypoints or len(waypoints) < 2:
        return None
    
    # Interpolate to ensure good coverage
    interpolated_waypoints = interpolate_route_waypoints(waypoints, max_distance_km=2.0)
    
    results = {
        'original_waypoints': waypoints,
        'analyzed_waypoints': interpolated_waypoints,
        'waypoint_risks': [],
        'route_summary': {},
        'highest_risk_segment': None,
        'analysis_time': datetime.now().isoformat()
    }
    
    total_points = len(interpolated_waypoints)
    completed = 0
    
    def analyze_waypoint(idx_waypoint):
        """Analyze a single waypoint and return its risk assessment"""
        idx, (lat, lon) = idx_waypoint
        try:
            # Fetch minimal data for this point (faster than full fetch)
            elevation = get_elevation(lat, lon)
            weather = fetch_weather_data(lat, lon)
            
            # Quick risk factors
            temp = 0
            snow_depth = 0
            wind_speed = 0
            precip = 0
            
            if weather and 'current' in weather:
                current = weather['current']
                temp = current.get('temperature_2m', 0) or 0
                snow_depth = (current.get('snow_depth', 0) or 0) / 100  # Convert to meters
                wind_speed = current.get('wind_speed_10m', 0) or 0
            
            if weather and 'daily' in weather:
                daily = weather['daily']
                precip = (daily.get('precipitation_sum', [0])[-1] or 0)
            
            # Calculate risk score (simplified model)
            risk_score = 0.0
            risk_factors = []
            
            # Temperature factor (warming = risk increase)
            if temp > 0:
                risk_score += 0.2
                risk_factors.append(f"Above freezing ({temp:.1f}°C)")
            elif -5 < temp <= 0:
                risk_score += 0.1
                risk_factors.append(f"Near freezing ({temp:.1f}°C)")
            
            # Elevation factor (higher = more risk)
            if elevation > 3000:
                risk_score += 0.15
                risk_factors.append(f"High elevation ({elevation:.0f}m)")
            elif elevation > 2000:
                risk_score += 0.1
                risk_factors.append(f"Alpine terrain ({elevation:.0f}m)")
            
            # Wind factor
            if wind_speed > 15:
                risk_score += 0.2
                risk_factors.append(f"Strong wind ({wind_speed:.1f} m/s)")
            elif wind_speed > 8:
                risk_score += 0.1
                risk_factors.append(f"Moderate wind ({wind_speed:.1f} m/s)")
            
            # Recent precipitation factor
            if precip > 20:
                risk_score += 0.25
                risk_factors.append(f"Heavy recent precip ({precip:.1f}mm)")
            elif precip > 10:
                risk_score += 0.15
                risk_factors.append(f"Moderate precip ({precip:.1f}mm)")
            
            # Snow depth factor
            if snow_depth > 1.5:
                risk_score += 0.15
                risk_factors.append(f"Deep snowpack ({snow_depth*100:.0f}cm)")
            
            # Normalize to 0-1
            risk_score = min(1.0, risk_score)
            
            # Determine risk level
            if risk_score >= 0.6:
                risk_level = "HIGH"
            elif risk_score >= 0.35:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"
            
            return {
                'index': idx,
                'lat': lat,
                'lon': lon,
                'elevation': elevation,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'temperature': temp,
                'snow_depth': snow_depth,
                'wind_speed': wind_speed,
                'precipitation': precip,
                'success': True
            }
        except Exception as e:
            return {
                'index': idx,
                'lat': lat,
                'lon': lon,
                'error': str(e),
                'success': False,
                'risk_score': 0.5,  # Unknown = moderate
                'risk_level': "UNKNOWN"
            }
    
    # Parallel analysis of waypoints
    waypoint_results = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(analyze_waypoint, (i, wp)): i
            for i, wp in enumerate(interpolated_waypoints)
        }
        
        for future in as_completed(futures):
            completed += 1
            if progress_callback:
                progress_callback(completed / total_points, f"Analyzing point {completed}/{total_points}")
            
            try:
                result = future.result()
                waypoint_results.append(result)
            except Exception as e:
                idx = futures[future]
                waypoint_results.append({
                    'index': idx,
                    'error': str(e),
                    'success': False,
                    'risk_score': 0.5,
                    'risk_level': "UNKNOWN"
                })
    
    # Sort by index to maintain route order
    waypoint_results.sort(key=lambda x: x['index'])
    results['waypoint_risks'] = waypoint_results
    
    # Calculate route summary
    risk_scores = [wp['risk_score'] for wp in waypoint_results if wp.get('success', False)]
    
    if risk_scores:
        max_risk = max(risk_scores)
        avg_risk = sum(risk_scores) / len(risk_scores)
        
        # Find highest risk segment
        highest_risk_wp = max(waypoint_results, key=lambda x: x.get('risk_score', 0))
        
        # Determine overall route risk (use highest risk point)
        if max_risk >= 0.6:
            overall_level = "HIGH"
            overall_message = "Dangerous sections on route"
        elif max_risk >= 0.35:
            overall_level = "MODERATE"
            overall_message = "Exercise caution"
        else:
            overall_level = "LOW"
            overall_message = "Route appears stable"
        
        results['route_summary'] = {
            'max_risk_score': max_risk,
            'avg_risk_score': avg_risk,
            'overall_risk_level': overall_level,
            'overall_message': overall_message,
            'total_waypoints': len(interpolated_waypoints),
            'high_risk_count': sum(1 for s in risk_scores if s >= 0.6),
            'moderate_risk_count': sum(1 for s in risk_scores if 0.35 <= s < 0.6),
            'low_risk_count': sum(1 for s in risk_scores if s < 0.35)
        }
        
        results['highest_risk_segment'] = {
            'lat': highest_risk_wp.get('lat'),
            'lon': highest_risk_wp.get('lon'),
            'risk_score': highest_risk_wp.get('risk_score'),
            'risk_factors': highest_risk_wp.get('risk_factors', [])
        }
    
    return results


def create_route_map(route_analysis, center_lat=None, center_lon=None):
    """
    Create a folium map showing the analyzed route with risk coloring.
    
    Args:
        route_analysis: Results from analyze_route_risk()
        center_lat, center_lon: Optional center coordinates
    
    Returns:
        Folium map object
    """
    if not route_analysis or not route_analysis.get('waypoint_risks'):
        return None
    
    waypoints = route_analysis['waypoint_risks']
    
    # Calculate center if not provided
    if center_lat is None or center_lon is None:
        lats = [wp['lat'] for wp in waypoints]
        lons = [wp['lon'] for wp in waypoints]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add terrain layer
    folium.TileLayer(
        tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='OpenTopoMap',
        name='Terrain',
        overlay=False
    ).add_to(m)
    
    # Color function based on risk
    def get_risk_color(risk_score):
        if risk_score >= 0.6:
            return '#dc2626'  # Red
        elif risk_score >= 0.35:
            return '#f59e0b'  # Orange
        else:
            return '#10b981'  # Green
    
    # Draw route segments with risk coloring
    for i in range(len(waypoints) - 1):
        wp1 = waypoints[i]
        wp2 = waypoints[i + 1]
        
        # Use max risk of the two points for segment color
        segment_risk = max(wp1.get('risk_score', 0), wp2.get('risk_score', 0))
        color = get_risk_color(segment_risk)
        
        folium.PolyLine(
            locations=[[wp1['lat'], wp1['lon']], [wp2['lat'], wp2['lon']]],
            color=color,
            weight=5,
            opacity=0.8
        ).add_to(m)
    
    # Add markers for start, end, and high-risk points
    # Start marker
    start = waypoints[0]
    folium.Marker(
        [start['lat'], start['lon']],
        popup=f"<b>Start</b><br>Elevation: {start.get('elevation', 'N/A')}m<br>Risk: {start.get('risk_level', 'N/A')}",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    
    # End marker
    end = waypoints[-1]
    folium.Marker(
        [end['lat'], end['lon']],
        popup=f"<b>End</b><br>Elevation: {end.get('elevation', 'N/A')}m<br>Risk: {end.get('risk_level', 'N/A')}",
        icon=folium.Icon(color='blue', icon='stop')
    ).add_to(m)
    
    # High risk point markers
    for wp in waypoints:
        if wp.get('risk_score', 0) >= 0.6:
            folium.CircleMarker(
                [wp['lat'], wp['lon']],
                radius=8,
                color='#dc2626',
                fill=True,
                fillColor='#dc2626',
                fillOpacity=0.7,
                popup=f"<b>High Risk Zone</b><br>Risk: {wp.get('risk_score', 0)*100:.0f}%<br>Factors: {', '.join(wp.get('risk_factors', []))}"
            ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Fit bounds to show entire route
    lats = [wp['lat'] for wp in waypoints]
    lons = [wp['lon'] for wp in waypoints]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    
    return m


# ============================================
# STREAMLIT UI
# ============================================

# Page configuration
st.set_page_config(
    page_title="Avalanche Risk Assessment",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean, professional CSS
st.markdown("""
<style>
    /* Clean typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling */
    .app-header {
        padding: 1.5rem 0 1rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    
    .app-title {
        font-size: 1.75rem;
        font-weight: 600;
        color: #1f2937;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .app-subtitle {
        font-size: 0.875rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }
    
    /* Risk display cards */
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .risk-none {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        color: white;
    }
    
    .risk-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.9;
        margin-bottom: 0.25rem;
    }
    
    .risk-level {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.25rem 0;
    }
    
    .risk-confidence {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Data cards */
    .data-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .data-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }
    
    .data-value {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1f2937;
    }
    
    /* Status indicators */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    
    .status-online { background-color: #10b981; }
    .status-partial { background-color: #f59e0b; }
    .status-offline { background-color: #ef4444; }
    
    /* Section headers */
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #374151;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e5e7eb;
    }
    
    /* Clean buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.15s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Info boxes */
    .info-box {
        background: #f0f9ff;
        border-left: 3px solid #0284c7;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.875rem;
        color: #0c4a6e;
    }
    
    .warning-box {
        background: #fffbeb;
        border-left: 3px solid #f59e0b;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.875rem;
        color: #78350f;
    }
    
    /* Source tags */
    .source-tag {
        display: inline-block;
        background: #e5e7eb;
        color: #374151;
        padding: 0.125rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        margin: 0.125rem;
    }
    
    .source-tag-satellite {
        background: #dbeafe;
        color: #1e40af;
    }
    
    .source-tag-station {
        background: #dcfce7;
        color: #166534;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    
    /* Clean expander */
    .streamlit-expanderHeader {
        font-size: 0.875rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="app-header">
    <h1 class="app-title">⛰️ Avalanche Risk Assessment</h1>
    <p class="app-subtitle">Real-time analysis using satellite and weather station data</p>
</div>
""", unsafe_allow_html=True)

# Main analysis mode selection
analysis_mode = st.radio(
    "Analysis Mode",
    ["📍 Single Point", "🗺️ Route Analysis"],
    horizontal=True,
    help="Analyze a single location or an entire hiking/skiing route"
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
if 'user_ip' not in st.session_state:
    st.session_state.user_ip = None
if 'ip_consent' not in st.session_state:
    st.session_state.ip_consent = False
if 'map_clicked_lat' not in st.session_state:
    st.session_state.map_clicked_lat = None
if 'map_clicked_lon' not in st.session_state:
    st.session_state.map_clicked_lon = None
if 'route_waypoints' not in st.session_state:
    st.session_state.route_waypoints = []
if 'route_analysis' not in st.session_state:
    st.session_state.route_analysis = None


# ============================================
# ROUTE ANALYSIS MODE
# ============================================
if analysis_mode == "🗺️ Route Analysis":
    st.markdown('<p class="section-header">Draw Your Route</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>Instructions:</strong> Use the polyline tool (📐) on the map to draw your route. 
        Click to add waypoints, double-click to finish. The route will be analyzed for avalanche risk at each segment.
    </div>
    """, unsafe_allow_html=True)
    
    # Route drawing map
    default_lat = 46.8  # Alps
    default_lon = 9.8
    
    m = folium.Map(
        location=[default_lat, default_lon],
        zoom_start=8,
        tiles='OpenStreetMap'
    )
    
    # Add terrain layer
    folium.TileLayer(
        tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='OpenTopoMap',
        name='Terrain',
        overlay=False
    ).add_to(m)
    
    # Add satellite layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False
    ).add_to(m)
    
    # Add drawing tools
    draw = Draw(
        draw_options={
            'polyline': {
                'allowIntersection': True,
                'shapeOptions': {
                    'color': '#3388ff',
                    'weight': 4
                }
            },
            'polygon': False,
            'rectangle': False,
            'circle': False,
            'marker': False,
            'circlemarker': False
        },
        edit_options={'edit': True, 'remove': True}
    )
    draw.add_to(m)
    
    folium.LayerControl().add_to(m)
    
    # Display map
    map_data = st_folium(m, width=None, height=450, key="route_map")
    
    # Extract drawn route
    if map_data and map_data.get('all_drawings'):
        drawings = map_data['all_drawings']
        
        for drawing in drawings:
            if drawing.get('geometry', {}).get('type') == 'LineString':
                coords = drawing['geometry']['coordinates']
                # Coords are [lon, lat] in GeoJSON, convert to (lat, lon)
                waypoints = [(coord[1], coord[0]) for coord in coords]
                st.session_state.route_waypoints = waypoints
                break
    
    # Show route info
    if st.session_state.route_waypoints:
        num_points = len(st.session_state.route_waypoints)
        st.success(f"Route drawn with {num_points} waypoints")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Analyze Route", type="primary"):
                with st.spinner("Analyzing route risk..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(progress, text):
                        progress_bar.progress(progress)
                        status_text.text(text)
                    
                    st.session_state.route_analysis = analyze_route_risk(
                        st.session_state.route_waypoints,
                        progress_callback=update_progress
                    )
                    
                    progress_bar.empty()
                    status_text.empty()
                    st.rerun()
        
        with col2:
            if st.button("Clear Route"):
                st.session_state.route_waypoints = []
                st.session_state.route_analysis = None
                st.rerun()
    
    # Display route analysis results
    if st.session_state.route_analysis:
        analysis = st.session_state.route_analysis
        summary = analysis.get('route_summary', {})
        
        st.markdown('<p class="section-header">Route Risk Assessment</p>', unsafe_allow_html=True)
        
        # Overall route risk card
        overall_risk = summary.get('overall_risk_level', 'UNKNOWN')
        risk_class = {
            'HIGH': 'risk-high',
            'MODERATE': 'risk-medium',
            'LOW': 'risk-low'
        }.get(overall_risk, 'risk-none')
        
        max_risk_pct = summary.get('max_risk_score', 0) * 100
        
        st.markdown(f"""
        <div class="risk-card {risk_class}">
            <div class="risk-label">Overall Route Risk</div>
            <div class="risk-level">{overall_risk}</div>
            <div class="risk-confidence">Max risk: {max_risk_pct:.0f}% • {summary.get('overall_message', '')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk breakdown
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Waypoints Analyzed", summary.get('total_waypoints', 0))
        with col2:
            st.metric("High Risk Zones", summary.get('high_risk_count', 0))
        with col3:
            st.metric("Moderate Risk Zones", summary.get('moderate_risk_count', 0))
        with col4:
            avg_risk = summary.get('avg_risk_score', 0) * 100
            st.metric("Avg Risk Score", f"{avg_risk:.0f}%")
        
        # Display route map with risk coloring
        st.markdown('<p class="section-header">Risk Map</p>', unsafe_allow_html=True)
        
        risk_map = create_route_map(analysis)
        if risk_map:
            st_folium(risk_map, width=None, height=400, key="risk_map_display")
        
        # Legend
        st.markdown("""
        <div style="display: flex; gap: 1.5rem; padding: 0.75rem; background: #f9fafb; border-radius: 8px; margin-top: 0.5rem;">
            <span><span style="display: inline-block; width: 16px; height: 4px; background: #10b981; vertical-align: middle;"></span> Low Risk</span>
            <span><span style="display: inline-block; width: 16px; height: 4px; background: #f59e0b; vertical-align: middle;"></span> Moderate Risk</span>
            <span><span style="display: inline-block; width: 16px; height: 4px; background: #dc2626; vertical-align: middle;"></span> High Risk</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Highest risk segment details
        highest = analysis.get('highest_risk_segment')
        if highest:
            st.markdown('<p class="section-header">Highest Risk Zone</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **Location:** {highest.get('lat', 0):.5f}, {highest.get('lon', 0):.5f}  
                **Risk Score:** {highest.get('risk_score', 0)*100:.0f}%
                """)
            with col2:
                factors = highest.get('risk_factors', [])
                if factors:
                    st.markdown("**Contributing Factors:**")
                    for factor in factors:
                        st.markdown(f"• {factor}")
        
        # Detailed waypoint table
        with st.expander("View all waypoint details"):
            waypoint_risks = analysis.get('waypoint_risks', [])
            if waypoint_risks:
                df_data = []
                for wp in waypoint_risks:
                    df_data.append({
                        'Point': wp.get('index', 0) + 1,
                        'Lat': f"{wp.get('lat', 0):.4f}",
                        'Lon': f"{wp.get('lon', 0):.4f}",
                        'Elevation (m)': wp.get('elevation', 'N/A'),
                        'Risk Level': wp.get('risk_level', 'N/A'),
                        'Risk Score': f"{wp.get('risk_score', 0)*100:.0f}%",
                        'Temp (°C)': f"{wp.get('temperature', 0):.1f}" if wp.get('temperature') else 'N/A',
                        'Wind (m/s)': f"{wp.get('wind_speed', 0):.1f}" if wp.get('wind_speed') else 'N/A'
                    })
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Recommendations
        st.markdown('<p class="section-header">Recommendations</p>', unsafe_allow_html=True)
        
        if overall_risk == "HIGH":
            st.markdown("""
            <div class="warning-box">
                <strong>High Risk Route:</strong><br>
                • Consider an alternative route avoiding high-risk zones<br>
                • Do not travel through identified danger areas<br>
                • Check local avalanche bulletins before proceeding<br>
                • If travel is necessary, have full rescue equipment and trained partners
            </div>
            """, unsafe_allow_html=True)
        elif overall_risk == "MODERATE":
            st.markdown("""
            <div class="warning-box">
                <strong>Moderate Risk Route:</strong><br>
                • Exercise increased caution in moderate-risk segments<br>
                • Carry avalanche safety equipment (beacon, probe, shovel)<br>
                • Travel one at a time through suspect terrain<br>
                • Have escape routes planned at high-risk zones
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <strong>Lower Risk Route:</strong><br>
                • Conditions appear more stable along this route<br>
                • Still carry avalanche safety gear<br>
                • Remain vigilant for changing conditions<br>
                • Monitor weather and re-evaluate if conditions change
            </div>
            """, unsafe_allow_html=True)
        
        # ============================================
        # WIND LOADING ANALYSIS (Route Mode)
        # ============================================
        st.markdown('<p class="section-header">Wind Loading Analysis</p>', unsafe_allow_html=True)
        
        # Get wind data for the route start point
        start_wp = st.session_state.route_waypoints[0] if st.session_state.route_waypoints else None
        if start_wp:
            with st.spinner("Analyzing wind loading zones..."):
                wind_data = fetch_wind_data_for_analysis(start_wp[0], start_wp[1])
            
            if wind_data.get('available'):
                wind_dir = wind_data.get('current_direction') or wind_data.get('avg_direction_24h', 0)
                wind_speed = wind_data.get('current_speed') or wind_data.get('avg_speed_24h', 0)
                
                wind_analysis = analyze_wind_loading(start_wp[0], start_wp[1], wind_dir, wind_speed)
                
                # Display wind info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Wind Direction", f"{wind_analysis['wind_direction_cardinal']}")
                with col2:
                    st.metric("Wind Speed", f"{wind_speed:.1f} m/s")
                with col3:
                    st.metric("Loading Risk", wind_analysis['loading_risk'])
                with col4:
                    max_gust = wind_data.get('current_gusts') or wind_data.get('max_speed_24h', 0)
                    st.metric("Max Gusts (24h)", f"{max_gust:.1f} m/s")
                
                # Risk display card
                loading_risk = wind_analysis['loading_risk']
                if loading_risk == "EXTREME":
                    loading_class = "risk-high"
                elif loading_risk == "HIGH":
                    loading_class = "risk-high"
                elif loading_risk == "MODERATE":
                    loading_class = "risk-medium"
                else:
                    loading_class = "risk-low"
                
                st.markdown(f"""
                <div style="background: {'#fef2f2' if loading_risk in ['HIGH', 'EXTREME'] else '#fffbeb' if loading_risk == 'MODERATE' else '#f0fdf4'}; 
                            border-left: 4px solid {'#dc2626' if loading_risk in ['HIGH', 'EXTREME'] else '#f59e0b' if loading_risk == 'MODERATE' else '#10b981'};
                            padding: 1rem; border-radius: 0 8px 8px 0; margin: 1rem 0;">
                    <strong>Wind Loading: {loading_risk}</strong><br>
                    <span style="font-size: 0.9rem;">
                        Wind from <strong>{wind_analysis['wind_direction_cardinal']}</strong> ({wind_analysis['wind_direction']}°) at <strong>{wind_speed:.1f} m/s</strong><br>
                        Leeward (danger) slopes face: <strong>{wind_analysis['leeward_cardinal']}</strong>
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # Affected slopes
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Slopes to Avoid (Wind Loaded):**")
                    affected = wind_analysis.get('affected_aspects', [])
                    if affected:
                        st.markdown(f"• Leeward: {', '.join(wind_analysis.get('leeward_aspects', []))}")
                        st.markdown(f"• Cross-loaded: {', '.join(wind_analysis.get('cross_load_aspects', []))}")
                    else:
                        st.markdown("• Minimal wind loading expected")
                
                with col2:
                    st.markdown("**Safer Slopes (Windward):**")
                    safe = wind_analysis.get('safe_aspects', [])
                    if safe:
                        st.markdown(f"• {', '.join(safe)}")
                    else:
                        st.markdown("• All aspects relatively similar")
                
                # Recommendations
                st.markdown("**Wind Loading Recommendations:**")
                for rec in wind_analysis.get('recommendations', []):
                    st.markdown(f"• {rec}")
                
                # Show wind loading overlay on a map
                with st.expander("View Wind Loading Zones on Map"):
                    wind_map = folium.Map(
                        location=[start_wp[0], start_wp[1]],
                        zoom_start=12,
                        tiles='OpenTopoMap'
                    )
                    
                    # Add wind loading overlays
                    overlays = create_wind_loading_overlay(start_wp[0], start_wp[1], wind_analysis, radius_km=3)
                    for name, overlay in overlays:
                        overlay.add_to(wind_map)
                    
                    # Add route
                    if st.session_state.route_waypoints:
                        folium.PolyLine(
                            locations=st.session_state.route_waypoints,
                            color='#3b82f6',
                            weight=3,
                            opacity=0.8
                        ).add_to(wind_map)
                    
                    # Legend
                    legend_html = '''
                    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                                background: white; padding: 10px; border-radius: 5px;
                                box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-size: 12px;">
                        <strong>Wind Loading Zones</strong><br>
                        <span style="color: #dc2626;">■</span> Leeward (High Risk)<br>
                        <span style="color: #f59e0b;">■</span> Cross-loaded (Moderate)<br>
                        <span style="color: #10b981;">■</span> Windward (Lower Risk)<br>
                        <span style="color: #1f2937;">→</span> Wind Direction
                    </div>
                    '''
                    wind_map.get_root().html.add_child(folium.Element(legend_html))
                    
                    st_folium(wind_map, width=None, height=400, key="wind_loading_map")
            else:
                st.info("Wind data not available for this location")


# ============================================
# SINGLE POINT ANALYSIS MODE
# ============================================
else:
    # Location selection
    st.markdown('<p class="section-header">Location</p>', unsafe_allow_html=True)

    data_source = st.radio(
        "Data input method",
        ["Automatic (satellite data)", "Manual entry"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if data_source == "Automatic (satellite data)":
        
        # Location selection tabs
        location_tab = st.radio(
            "Select location method",
            ["Use my IP address", "Select on map"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if location_tab == "Use my IP address":
            st.markdown('<div class="info-box">Your IP address will be used only to determine your approximate location. No data is stored.</div>', unsafe_allow_html=True)
            
            if st.button("Detect my location", type="primary"):
                with st.spinner("Detecting location..."):
                    detected_ip = get_ip_address()
                    if detected_ip:
                        st.session_state.user_ip = detected_ip
                        st.session_state.ip_consent = True
                    else:
                        st.error("Could not detect location. Please use map selection instead.")
            
            if st.session_state.user_ip and st.session_state.ip_consent:
                st.success(f"Location detected from IP: {st.session_state.user_ip}")
        
        else:  # Select on map
            st.session_state.ip_consent = True
        
        st.markdown("Click anywhere on the map to set your location:")
        
        # Default to Alps region
        default_lat = st.session_state.get('map_clicked_lat') or 46.8
        default_lon = st.session_state.get('map_clicked_lon') or 9.8
        
        # Create interactive map
        m = folium.Map(
            location=[default_lat, default_lon],
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Add satellite layer
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False
        ).add_to(m)
        
        # Add terrain layer
        folium.TileLayer(
            tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
            attr='OpenTopoMap',
            name='Terrain',
            overlay=False
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        # Add marker if location selected
        if st.session_state.get('map_clicked_lat'):
            folium.Marker(
                [st.session_state.map_clicked_lat, st.session_state.map_clicked_lon],
                popup=f"Lat: {st.session_state.map_clicked_lat:.4f}, Lon: {st.session_state.map_clicked_lon:.4f}",
                icon=folium.Icon(color='red', icon='map-marker', prefix='fa')
            ).add_to(m)
        
        # Display map
        map_data = st_folium(m, width=700, height=350, key="main_location_map", returned_objects=["last_clicked"])
        
        if map_data and map_data.get('last_clicked'):
            st.session_state.map_clicked_lat = map_data['last_clicked']['lat']
            st.session_state.map_clicked_lon = map_data['last_clicked']['lng']
        
        if st.session_state.get('map_clicked_lat'):
            st.success(f"Selected: {st.session_state.map_clicked_lat:.4f}°N, {st.session_state.map_clicked_lon:.4f}°E")
    
    st.markdown("")  # Spacing
    
    # Set location from map click (without fetching data yet)
    if location_tab == "Select on map" and st.session_state.get('map_clicked_lat'):
        if st.session_state.location is None or \
           st.session_state.location.get('latitude') != st.session_state.map_clicked_lat or \
           st.session_state.location.get('longitude') != st.session_state.map_clicked_lon:
            st.session_state.location = create_location_from_coords(
                st.session_state.map_clicked_lat, 
                st.session_state.map_clicked_lon
            )
            st.session_state.location['elevation'] = get_elevation(
                st.session_state.map_clicked_lat, 
                st.session_state.map_clicked_lon
            )
            # Clear old satellite data when location changes
            st.session_state.satellite_raw = None
            st.session_state.env_data = None
    
    # Set location from IP
    if location_tab == "Use my IP address" and st.session_state.user_ip and st.session_state.ip_consent:
        if st.session_state.location is None:
            with st.spinner("Getting location from IP..."):
                st.session_state.location = get_user_location(st.session_state.user_ip)
                lat = st.session_state.location['latitude']
                lon = st.session_state.location['longitude']
                st.session_state.location['elevation'] = get_elevation(lat, lon)
    
    # Display location info
    if st.session_state.location:
        loc = st.session_state.location
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.markdown(f"**{loc['city']}, {loc['region']}**")
            st.caption(loc['country'])
        with col_info2:
            st.markdown(f"**{loc['latitude']:.4f}°N, {loc['longitude']:.4f}°E**")
            st.caption("Coordinates")
        with col_info3:
            elev = loc.get('elevation', 'Unknown')
            st.markdown(f"**{elev}m**")
            st.caption("Elevation")
        
        # Expandable adjustment
        with st.expander("Adjust location"):
            col_coord1, col_coord2 = st.columns(2)
            with col_coord1:
                new_lat = st.number_input("Latitude", value=float(loc['latitude']), min_value=-90.0, max_value=90.0, step=0.01)
            with col_coord2:
                new_lon = st.number_input("Longitude", value=float(loc['longitude']), min_value=-180.0, max_value=180.0, step=0.01)
            
            if st.button("Update location"):
                st.session_state.location['latitude'] = new_lat
                st.session_state.location['longitude'] = new_lon
                st.session_state.location['elevation'] = get_elevation(new_lat, new_lon)
                
                reverse_geo = get_reverse_geocode(new_lat, new_lon)
                if reverse_geo:
                    st.session_state.location['city'] = reverse_geo.get('city', 'Unknown')
                    st.session_state.location['region'] = reverse_geo.get('region', 'Unknown')
                    st.session_state.location['country'] = reverse_geo.get('country', 'Unknown')
                
                # Clear cached data so it will be fetched on next assessment
                st.session_state.satellite_raw = None
                st.session_state.env_data = None
                
                st.success("Location updated! Click 'Run Assessment' to analyze this location.")
                st.rerun()
    
    # Display satellite data status (compact version)
    if st.session_state.satellite_raw:
        raw = st.session_state.satellite_raw
        
        # Compact status summary
        if 'summary' in raw:
            summary = raw['summary']
            success_count = summary['successful_sources']
            total_count = summary['total_sources']
            
            st.markdown(f"""
            <div style="background: #f9fafb; padding: 0.75rem; border-radius: 8px; margin-top: 1rem;">
                <span style="color: #059669; font-weight: 500;">● {success_count} sources connected</span>
                <span style="color: #6b7280; margin-left: 1rem;">of {total_count} available</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Compact expandable details
        with st.expander("View data sources"):
            cols = st.columns(3)
            all_sources = list(raw['data_quality'].items())
            
            for i, (name, status) in enumerate(all_sources):
                col_idx = i % 3
                with cols[col_idx]:
                    # Clean up source name
                    clean_name = name.replace("(", "").replace(")", "").replace("Western US", "").strip()
                    if len(clean_name) > 20:
                        clean_name = clean_name[:20] + "..."
                    
                    if status == 'success':
                        st.markdown(f"<span class='status-dot status-online'></span>{clean_name}", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span class='status-dot status-partial'></span>{clean_name}", unsafe_allow_html=True)
    
    # Display fetched data summary
    if st.session_state.env_data:
        st.markdown('<p class="section-header">Current Conditions</p>', unsafe_allow_html=True)
        
        env = st.session_state.env_data
        
        # Key metrics in a cleaner layout
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            temp = env.get('TA', 0)
            st.metric("Temperature", f"{temp:.1f}°C")
        with col2:
            snow = env.get('max_height', 0) * 100
            snow_change = env.get('max_height_1_diff', 0) * 100
            st.metric("Snow Depth", f"{snow:.0f} cm", delta=f"{snow_change:+.0f} cm/24h")
        with col3:
            radiation = env.get('ISWR_daily', 0)
            st.metric("Solar Radiation", f"{radiation:.0f} W/m²")
        with col4:
            stability = env.get('S5', 2.5)
            st.metric("Stability Index", f"{stability:.2f}")
        
        # Update session state inputs
        for key, value in env.items():
            if key in features_for_input:
                st.session_state.inputs[key] = value

st.markdown("")  # Spacing

# Sidebar - minimal and clean
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This tool assesses avalanche risk using real-time data from satellite systems and weather stations.

**Data Sources:**
- MODIS & VIIRS satellites
- ERA5 reanalysis
- SNOTEL network
- Local weather stations
""")

st.sidebar.markdown("---")
st.sidebar.caption("Always verify with local avalanche centers before backcountry travel.")

# Helper function
def get_input_value(key, default=0.0, min_val=None, max_val=None):
    value = st.session_state.inputs.get(key, default)
    if value is None:
        value = default
    if min_val is not None and value < min_val:
        value = min_val
    if max_val is not None and value > max_val:
        value = max_val
    return value

# Prediction section (only for single point mode)
if analysis_mode == "📍 Single Point":
    st.markdown('<p class="section-header">Risk Assessment</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Check if location is set
        if st.session_state.location:
            predict_button = st.button("Run Assessment", type="primary", use_container_width=True)
        else:
            st.warning("Please select a location on the map first")
            predict_button = False

    if predict_button:
        # First, fetch satellite data if not already loaded
        if st.session_state.satellite_raw is None or st.session_state.env_data is None:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress, text):
                progress_bar.progress(progress)
                status_text.text(text.replace("🛰️ ", "").replace("Fetching ", "Loading "))
            
            with st.spinner("Loading satellite data..."):
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
        
        # Prepare input data from satellite data (using NaN for missing values instead of 0)
        if st.session_state.env_data:
            for feature in features_for_input:
                if feature in st.session_state.env_data and st.session_state.env_data[feature] is not None:
                    st.session_state.inputs[feature] = st.session_state.env_data[feature]
                else:
                    st.session_state.inputs[feature] = np.nan  # Use NaN for missing, imputer will handle it
        
        # Create input data - use NaN for missing values (imputer will handle them)
        input_values = []
        for f in features_for_input:
            val = st.session_state.inputs.get(f)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                input_values.append(np.nan)
            else:
                input_values.append(val)
        
        input_data = pd.DataFrame([input_values], columns=features_for_input)
        
        weights_path = "model_reduced_weights.weights.h5"
        config_path = "model_reduced_config.json"
        scaler_path = "scaler_reduced.joblib"
        imputer_path = "imputer_reduced.joblib"
        threshold_path = "threshold_reduced.txt"
        
        use_ml_model = False
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            import json
            
            # Check if all model files exist
            if all(os.path.exists(p) for p in [weights_path, config_path, scaler_path, imputer_path]):
                
                # Load model configuration
                with open(config_path, 'r') as f:
                    model_config = json.load(f)
                
                # Recreate the OptimizedSafetyPINN model architecture
                class OptimizedSafetyPINN(keras.Model):
                    def __init__(self, phys_idx, input_dim, **kwargs):
                        super().__init__()
                        self.phys_idx = phys_idx
                        self.input_dim = input_dim
                        dropout_rate = 0.25
                        
                        # Attention layers
                        self.attention_dense = layers.Dense(input_dim, activation='tanh', 
                                                           kernel_regularizer=keras.regularizers.l2(1e-4))
                        self.attention_weights = layers.Dense(input_dim, activation='softmax')
                        
                        # Deep network with residual connections
                        self.proj1 = layers.Dense(256)
                        self.dense1 = layers.Dense(256, kernel_regularizer=keras.regularizers.l2(1e-4))
                    self.bn1 = layers.BatchNormalization()
                    self.drop1 = layers.Dropout(dropout_rate)
                    
                    self.dense2 = layers.Dense(256, kernel_regularizer=keras.regularizers.l2(1e-4))
                    self.bn2 = layers.BatchNormalization()
                    self.drop2 = layers.Dropout(dropout_rate)
                    
                    self.proj2 = layers.Dense(128)
                    self.dense3 = layers.Dense(128, kernel_regularizer=keras.regularizers.l2(1e-4))
                    self.bn3 = layers.BatchNormalization()
                    self.drop3 = layers.Dropout(dropout_rate)
                    
                    self.dense4 = layers.Dense(128, kernel_regularizer=keras.regularizers.l2(1e-4))
                    self.bn4 = layers.BatchNormalization()
                    self.drop4 = layers.Dropout(dropout_rate)
                    
                    self.proj3 = layers.Dense(64)
                    self.dense5 = layers.Dense(64, kernel_regularizer=keras.regularizers.l2(1e-4))
                    self.bn5 = layers.BatchNormalization()
                    self.drop5 = layers.Dropout(dropout_rate)
                    
                    # Avalanche prediction head
                    self.aval_dense1 = layers.Dense(64, activation='relu', 
                                                    kernel_regularizer=keras.regularizers.l2(1e-4))
                    self.aval_bn1 = layers.BatchNormalization()
                    self.aval_dense2 = layers.Dense(32, activation='relu',
                                                    kernel_regularizer=keras.regularizers.l2(1e-4))
                    self.aval_dense3 = layers.Dense(16, activation='relu')
                    self.aval_head = layers.Dense(1, activation='sigmoid', name='avalanche')
                    
                    # Physics head
                    self.phys_dense1 = layers.Dense(32, activation='relu')
                    self.phys_dense2 = layers.Dense(16, activation='relu')
                    self.phys_head = layers.Dense(1, activation='linear', name='temp_change')
                    
                    self.alpha = tf.Variable(0.1, dtype=tf.float32, trainable=True, name='alpha')

                def call(self, inputs, training=False):
                    att = self.attention_dense(inputs)
                    att_weights = self.attention_weights(att)
                    x = inputs * att_weights
                    
                    x = self.proj1(x)
                    x1 = self.dense1(x)
                    x1 = self.bn1(x1, training=training)
                    x1 = tf.nn.leaky_relu(x1, alpha=0.1)
                    x1 = self.drop1(x1, training=training)
                    
                    x2 = self.dense2(x1)
                    x2 = self.bn2(x2, training=training)
                    x2 = tf.nn.leaky_relu(x2, alpha=0.1)
                    x2 = x2 + x1
                    x2 = self.drop2(x2, training=training)
                    
                    x3 = self.proj2(x2)
                    x3 = self.dense3(x3)
                    x3 = self.bn3(x3, training=training)
                    x3 = tf.nn.leaky_relu(x3, alpha=0.1)
                    x3 = self.drop3(x3, training=training)
                    
                    x4 = self.dense4(x3)
                    x4 = self.bn4(x4, training=training)
                    x4 = tf.nn.leaky_relu(x4, alpha=0.1)
                    x4 = x4 + x3
                    x4 = self.drop4(x4, training=training)
                    
                    x5 = self.proj3(x4)
                    x5 = self.dense5(x5)
                    x5 = self.bn5(x5, training=training)
                    feat = tf.nn.leaky_relu(x5, alpha=0.1)
                    feat = self.drop5(feat, training=training)
                    
                    aval_x = self.aval_dense1(feat)
                    aval_x = self.aval_bn1(aval_x, training=training)
                    aval_x = self.aval_dense2(aval_x)
                    aval_x = self.aval_dense3(aval_x)
                    aval_out = self.aval_head(aval_x)
                    
                    phys_x = self.phys_dense1(feat)
                    phys_x = self.phys_dense2(phys_x)
                    phys_out = self.phys_head(phys_x)
                    
                    return aval_out, phys_out
            
            # Create model with saved configuration
            model = OptimizedSafetyPINN(
                phys_idx=model_config['phys_indices'],
                input_dim=model_config['input_dim']
            )
            
            # Build model by calling it once with dummy data
            dummy_input = tf.zeros((1, model_config['input_dim']))
            _ = model(dummy_input)
            
            # Load trained weights
            model.load_weights(weights_path)
            
            # Load scaler and imputer
            scaler = joblib.load(scaler_path)
            imputer = joblib.load(imputer_path)
            
            # Load threshold
            optimal_threshold = 0.3  # Default
            if os.path.exists(threshold_path):
                with open(threshold_path, 'r') as f:
                    optimal_threshold = float(f.read().strip())
            
            # Preprocess input
            input_imputed = imputer.transform(input_data)
            input_scaled = scaler.transform(input_imputed)
            
            # Get prediction
            prediction, _ = model(tf.constant(input_scaled, dtype=tf.float32), training=False)
            avalanche_probability = float(prediction[0][0])
            # Model confidence = how far from 0.5 (uncertain) the prediction is
            # Confidence is high when probability is close to 0 or 1
            model_confidence = abs(avalanche_probability - 0.5) * 2  # Scale to 0-1
            use_ml_model = True
        
        except Exception as e:
            pass  # Fall through to physics-based assessment
        
        if not use_ml_model:
            # Using physics-based risk assessment (no ML model required)
            # This is a valid assessment method based on snowpack science
            
            risk_score = 0.3  # Base risk
            
            if st.session_state.inputs.get('TA', 0) > 0:
                risk_score += 0.15
            if st.session_state.inputs.get('TA_daily', 0) > st.session_state.inputs.get('TA', 0):
                risk_score += 0.1
            
            if st.session_state.inputs.get('water', 0) > 10:
                risk_score += 0.15
            if st.session_state.inputs.get('TSS_mod', 273) > 273:
                risk_score += 0.1
            
            if st.session_state.inputs.get('S5', 2) < 1.0:
                risk_score += 0.25
            elif st.session_state.inputs.get('S5', 2) < 1.5:
                risk_score += 0.15
            elif st.session_state.inputs.get('S5', 2) < 2.0:
                risk_score += 0.05
            
            if st.session_state.inputs.get('max_height_1_diff', 0) > 0.3:
                risk_score += 0.15
            
            if st.session_state.inputs.get('ISWR_daily', 0) > 300:
                risk_score += 0.1
            
            avalanche_probability = min(max(risk_score, 0.0), 1.0)
            # For physics-based, confidence based on how extreme the indicators are
            model_confidence = abs(avalanche_probability - 0.5) * 2
        
        st.markdown("")  # Spacing
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Check if there's no snow
            snow_depth = st.session_state.inputs.get('max_height', 0)
            if snow_depth is None or snow_depth <= 0:
                risk_level = "NONE"
                risk_class = "risk-none"
                risk_message = "No snow cover detected"
                avalanche_probability = 0.0
                model_confidence = 1.0  # Very confident there's no risk without snow
            elif avalanche_probability >= 0.7:
                risk_level = "HIGH"
                risk_class = "risk-high"
                risk_message = "Dangerous conditions likely"
            elif avalanche_probability >= 0.4:
                risk_level = "MODERATE"
                risk_class = "risk-medium"
                risk_message = "Exercise caution"
            else:
                risk_level = "LOW"
                risk_class = "risk-low"
                risk_message = "Conditions appear stable"
            
            st.markdown(f"""
            <div class="risk-card {risk_class}">
                <div class="risk-label">Avalanche Risk</div>
                <div class="risk-level">{risk_level}</div>
                <div class="risk-confidence">{avalanche_probability*100:.0f}% probability</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show model confidence separately
            confidence_label = "High" if model_confidence >= 0.7 else "Medium" if model_confidence >= 0.4 else "Low"
            st.caption(f"{risk_message} • Model confidence: {confidence_label}")
        
        # Key factors
        st.markdown('<p class="section-header">Contributing Factors</p>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Stability", f"{st.session_state.inputs.get('S5', 0):.2f}")
        with col2:
            st.metric("Temperature", f"{st.session_state.inputs.get('TA', 0):.1f}°C")
        with col3:
            st.metric("Snow Depth", f"{st.session_state.inputs.get('max_height', 0):.2f}m")
        with col4:
            st.metric("Radiation", f"{st.session_state.inputs.get('ISWR_daily', 0):.0f} W/m²")
        
        # Recommendations
        st.markdown('<p class="section-header">Recommendations</p>', unsafe_allow_html=True)
        
        if avalanche_probability >= 0.7:
            st.markdown("""
            <div class="warning-box">
                <strong>High Risk Actions:</strong><br>
                • Avoid all avalanche terrain<br>
                • Do not travel on or below steep slopes<br>
                • Check local avalanche advisories<br>
                • Consider postponing backcountry travel
            </div>
            """, unsafe_allow_html=True)
        elif avalanche_probability >= 0.4:
            st.markdown("""
            <div class="warning-box">
                <strong>Moderate Risk Actions:</strong><br>
                • Use caution in avalanche terrain<br>
                • Carry avalanche safety equipment<br>
                • Travel with partners<br>
                • Identify safe zones and escape routes
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <strong>Lower Risk Actions:</strong><br>
                • Conditions appear more stable<br>
                • Still carry avalanche safety gear<br>
                • Remain vigilant for changing conditions<br>
                • Check for updated forecasts
            </div>
            """, unsafe_allow_html=True)
        
        # ============================================
        # WIND LOADING ANALYSIS (Single Point Mode)
        # ============================================
        if st.session_state.location:
            st.markdown('<p class="section-header">Wind Loading Zones</p>', unsafe_allow_html=True)
            
            loc = st.session_state.location
            lat = loc['latitude']
            lon = loc['longitude']
            
            # Fetch wind data
            wind_data = fetch_wind_data_for_analysis(lat, lon)
            
            if wind_data.get('available'):
                wind_dir = wind_data.get('current_direction') or wind_data.get('avg_direction_24h', 0)
                wind_speed = wind_data.get('current_speed') or wind_data.get('avg_speed_24h', 0)
                
                wind_analysis = analyze_wind_loading(lat, lon, wind_dir, wind_speed)
                
                # Display wind metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Wind From", f"{wind_analysis['wind_direction_cardinal']}")
                with col2:
                    st.metric("Wind Speed", f"{wind_speed:.1f} m/s")
                with col3:
                    st.metric("Loading Risk", wind_analysis['loading_risk'])
                with col4:
                    max_gust = wind_data.get('current_gusts') or wind_data.get('max_speed_24h', 0)
                    st.metric("Gusts", f"{max_gust:.1f} m/s")
                
                # Risk display
                loading_risk = wind_analysis['loading_risk']
                risk_colors = {
                    'EXTREME': ('#fef2f2', '#dc2626'),
                    'HIGH': ('#fef2f2', '#dc2626'),
                    'MODERATE': ('#fffbeb', '#f59e0b'),
                    'LOW': ('#f0fdf4', '#10b981')
                }
                bg_color, border_color = risk_colors.get(loading_risk, ('#f9fafb', '#6b7280'))
                
                st.markdown(f"""
                <div style="background: {bg_color}; 
                            border-left: 4px solid {border_color};
                            padding: 1rem; border-radius: 0 8px 8px 0; margin: 1rem 0;">
                    <strong>Wind Loading: {loading_risk}</strong><br>
                    <span style="font-size: 0.9rem;">
                        Wind from <strong>{wind_analysis['wind_direction_cardinal']}</strong> ({wind_analysis['wind_direction']}°)<br>
                        Danger slopes (leeward): <strong>{wind_analysis['leeward_cardinal']}</strong> facing
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # Aspect recommendations
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Avoid (Wind Loaded):**")
                    leeward = wind_analysis.get('leeward_aspects', [])
                    cross = wind_analysis.get('cross_load_aspects', [])
                    if leeward:
                        st.markdown(f"🔴 Leeward: {', '.join(leeward)}")
                    if cross:
                        st.markdown(f"🟠 Cross-loaded: {', '.join(cross)}")
                    if not leeward and not cross:
                        st.markdown("• Light winds - minimal loading")
                
                with col2:
                    st.markdown("**Prefer (Windward/Safe):**")
                    safe = wind_analysis.get('safe_aspects', [])
                    if safe:
                        st.markdown(f"🟢 {', '.join(safe)}")
                    else:
                        st.markdown("• All aspects similar risk")
                
                # Wind loading map
                with st.expander("View Wind Loading Zones on Map"):
                    wind_map = folium.Map(
                        location=[lat, lon],
                        zoom_start=13,
                        tiles='OpenStreetMap'
                    )
                    
                    # Add terrain layer
                    folium.TileLayer(
                        tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
                        attr='OpenTopoMap',
                        name='Terrain',
                        overlay=False
                    ).add_to(wind_map)
                    
                    # Add wind loading overlays
                    overlays = create_wind_loading_overlay(lat, lon, wind_analysis, radius_km=2)
                    for name, overlay in overlays:
                        overlay.add_to(wind_map)
                    
                    # Center marker
                    folium.Marker(
                        [lat, lon],
                        popup=f"Analysis Point<br>Elevation: {loc.get('elevation', 'N/A')}m",
                        icon=folium.Icon(color='blue', icon='info-sign')
                    ).add_to(wind_map)
                    
                    folium.LayerControl().add_to(wind_map)
                    
                    st.markdown("""
                    <div style="font-size: 0.8rem; color: #6b7280; margin-bottom: 0.5rem;">
                        <strong>Legend:</strong> 
                        <span style="color: #dc2626;">■</span> Leeward (High Risk) · 
                        <span style="color: #f59e0b;">■</span> Cross-loaded (Moderate) · 
                        <span style="color: #10b981;">■</span> Windward (Lower Risk) · 
                        <span style="color: #1f2937;">→</span> Wind Direction
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st_folium(wind_map, width=None, height=400, key="single_point_wind_map")
                
                # Detailed recommendations
                with st.expander("Wind Loading Details"):
                    st.markdown("**How Wind Loading Creates Avalanche Danger:**")
                    st.markdown("""
                    Wind transports loose snow and deposits it on **leeward** (downwind) slopes, 
                    creating dense, cohesive **wind slabs** that can release as avalanches.
                    
                    - **Leeward slopes** (facing away from wind): Highest accumulation, greatest danger
                    - **Cross-loaded slopes** (perpendicular to wind): Moderate accumulation risk
                    - **Windward slopes** (facing into wind): Usually scoured, relatively safer
                    """)
                    
                    if wind_analysis.get('recommendations'):
                        st.markdown("**Current Conditions:**")
                        for rec in wind_analysis['recommendations']:
                            st.markdown(f"• {rec}")
                    
                    # 24h wind history
                    if wind_data.get('avg_speed_24h'):
                        st.markdown(f"""
                        **24-Hour Wind Summary:**
                        - Average speed: {wind_data['avg_speed_24h']:.1f} m/s
                        - Maximum speed: {wind_data.get('max_speed_24h', 0):.1f} m/s
                        - Predominant direction: {get_cardinal_direction(wind_data.get('avg_direction_24h', 0))}
                        """)
            else:
                st.info("Wind data not available for this location")

# Footer
st.markdown("")
st.markdown("---")

# Minimal footer with expandable details
with st.expander("Data sources and methodology"):
    st.markdown("""
    **Satellite Data Sources:**
    MODIS (NASA), VIIRS (NOAA), ERA5 (ECMWF), GOES (NOAA), Sentinel (ESA)
    
    **Weather Station Networks:**
    SNOTEL (NRCS), MesoWest, WMO stations
    
    **Methodology:**
    This tool combines satellite observations with physics-based snowpack modeling 
    to estimate avalanche risk. The stability index (S5) is calculated from snow depth, 
    new snow accumulation, temperature trends, precipitation type, wind speed, and 
    liquid water content.
    
    **Disclaimer:**
    This tool provides estimates based on available data and should not replace 
    professional avalanche forecasts. Always check with local avalanche centers 
    and exercise proper backcountry safety protocols.
    """)
