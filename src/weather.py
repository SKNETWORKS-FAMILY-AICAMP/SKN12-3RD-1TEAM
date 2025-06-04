import os
import json
import math
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()
open_data = os.getenv('OPEN_DATA')

def get_current_time() -> Dict[str, str]:
    """
    현재 시간 정보를 반환합니다.
    Returns:
        Dict with current date and time information
    """
    now = datetime.now()
    return {
        'date': now.strftime('%Y년 %m월 %d일'),
        'time': now.strftime('%H:%M'),
        'full_datetime': now.strftime('%Y년 %m월 %d일 %H:%M')
    }

# 강수형태 코드 매핑 (기상청)
PTY_MAP = {
    '0': '맑음',
    '1': '비',
    '2': '비/눈',
    '3': '눈',
    '4': '소나기'
}

# 위경도 → nx, ny 변환 함수
def latlon_to_grid(lat, lon):
    """
    Converts latitude/longitude to KMA grid coordinates (nx, ny).
    """
    RE = 6371.00877  # Earth radius (km)
    GRID = 5.0       # Grid spacing (km)
    SLAT1 = 30.0     # Projection latitude 1 (degree)
    SLAT2 = 60.0     # Projection latitude 2 (degree)
    OLON = 126.0     # Reference longitude (degree)
    OLAT = 38.0      # Reference latitude (degree)
    XO = 43          # Reference point X (GRID)
    YO = 136         # Reference point Y (GRID)
    DEGRAD = math.pi / 180.0
    re = RE / GRID
    slat1 = SLAT1 * DEGRAD
    slat2 = SLAT2 * DEGRAD
    olon = OLON * DEGRAD
    olat = OLAT * DEGRAD
    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = math.pow(sf, sn) * math.cos(slat1) / sn
    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = re * sf / math.pow(ro, sn)
    ra = math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5)
    ra = re * sf / math.pow(ra, sn)
    theta = lon * DEGRAD - olon
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn
    x = ra * math.sin(theta) + XO + 0.5
    y = ro - ra * math.cos(theta) + YO + 0.5
    return int(x), int(y)

# 기상청 API용 날짜/시간 계산
def get_base_date_time():
    now = datetime.now()
    base_date = now.strftime('%Y%m%d')
    base_time = now.strftime('%H') + '00'
    if now.minute < 40:
        prev_hour = (now - timedelta(hours=1)).strftime('%H')
        base_time = prev_hour + '00'
    return base_date, base_time

# 날씨 정보 조회 함수
def get_weather(city, city_info_path='../data/json/region_only_city_info.json'):
    """
    Returns weather info for a given region name (e.g. '서울', '부산', '수원', '기장').
    Finds the first key in city_info JSON that contains the input string if exact match is not found.
    Returns dict: {city, temperature, humidity, precipitation_type, wind_speed}
    If an error occurs, returns dict with 'error' key and message.
    """
    try:
        with open(city_info_path, 'r', encoding='utf-8') as f:
            city_data = json.load(f)
    except FileNotFoundError:
        return {'error': f"city info file not found: {city_info_path}"}
    except json.JSONDecodeError:
        return {'error': f"city info file is not valid JSON: {city_info_path}"}

    region_key = city if city in city_data else next((k for k in city_data if city in k), None)
    if not region_key:
        return {'error': f"'{city}'에 해당하는 지역이 region_only_city_info.json에 없습니다."}
    try:
        lat = float(city_data[region_key]['lat'])
        lon = float(city_data[region_key]['lon'])
    except (KeyError, ValueError, TypeError):
        return {'error': f"Invalid lat/lon for region: {region_key}"}
    nx, ny = latlon_to_grid(lat, lon)
    base_date, base_time = get_base_date_time()
    url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst'
    params = {
        'serviceKey': open_data,
        'pageNo': '1',
        'numOfRows': '1000',
        'dataType': 'XML',
        'base_date': base_date,
        'base_time': base_time,
        'nx': str(nx),
        'ny': str(ny)
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return {'error': f"Weather API request failed: {e}"}
    try:
        root = ET.fromstring(response.content)
        weather = {item.find('category').text: item.find('obsrValue').text for item in root.iter('item')}
    except ET.ParseError:
        return {'error': "Weather API response is not valid XML."}
    except Exception as e:
        return {'error': f"Error parsing weather data: {e}"}
    if not weather:
        return {'error': "No weather data found for the given region."}
    return {
        'city': region_key,
        'temperature': weather.get('T1H'),
        'humidity': weather.get('REH'),
        'precipitation_type': PTY_MAP.get(weather.get('PTY', '0'), '알수없음'),
        'wind_speed': weather.get('WSD')
    }
    
