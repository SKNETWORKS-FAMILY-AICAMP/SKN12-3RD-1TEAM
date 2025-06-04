#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from pprint import pprint

# 프로젝트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# weather 모듈 import
from weather import get_weather

def test_weather_function():
    """날씨 함수 테스트"""
    print("=== 날씨 함수 테스트 ===")
    
    # 테스트할 도시들
    test_cities = ["속초", "속초시", "서울", "부산"]
    
    for city in test_cities:
        print(f"\n🌤️  {city} 날씨 조회 중...")
        
        # 절대 경로로 JSON 파일 지정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, '../data', 'json', 'region_only_city_info.json')
        
        # 날씨 정보 가져오기
        weather_result = get_weather(city, city_info_path=json_path)
        
        print(f"결과: ")
        pprint(weather_result)
        
        # 오류 체크
        if 'error' in weather_result:
            print(f"❌ 오류 발생: {weather_result['error']}")
        else:
            print(f"✅ 성공: {city}의 날씨 정보를 가져왔습니다")

def test_different_city_formats():
    """다양한 도시명 형태 테스트"""
    print("\n=== 다양한 도시명 형태 테스트 ===")
    
    # 속초 관련 다양한 형태
    city_variations = ["속초", "속초시", "강원 속초", "강원도 속초시"]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, '../data', 'json', 'region_only_city_info.json')
    
    for city in city_variations:
        print(f"\n🔍 '{city}' 검색 중...")
        weather_result = get_weather(city, city_info_path=json_path)
        
        if 'error' in weather_result:
            print(f"❌ 실패: {weather_result['error']}")
        else:
            print(f"✅ 성공: 인식된 지역 -> {weather_result.get('city', 'Unknown')}")
            print(f"   기온: {weather_result.get('temperature', 'N/A')}°C")
            print(f"   습도: {weather_result.get('humidity', 'N/A')}%")
            print(f"   강수: {weather_result.get('precipitation_type', 'N/A')}")

if __name__ == "__main__":
    test_weather_function()
    test_different_city_formats() 