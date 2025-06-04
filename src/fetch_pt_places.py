import subprocess
import json
import os 
from typing import List, Dict, Optional
from dotenv import load_dotenv 

# load key 
load_dotenv()
service_key = os.getenv('TOUR_API_KEY')

def fetch_area_items(area_code: Optional[int] = None) -> List[Dict]:
    url = (
        f"https://apis.data.go.kr/B551011/KorPetTourService/areaCode"
        f"?serviceKey={service_key}&MobileOS=ETC&MobileApp=TestApp&_type=json&numOfRows=100"
    )
    if area_code:
        url += f"&areaCode={area_code}"
    result = subprocess.run(['curl', '-s', url], capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return data["response"]["body"]["items"]["item"]

def match_region_to_codes(region: str) -> (Optional[int], Optional[int]):
    """지역명을 시/도 및 시/군/구로 매핑"""
    try:
        # 1. 시/도 목록에서 일치 또는 포함되는 지역 찾기
        area_items = fetch_area_items()
        for item in area_items:
            if region in item["name"] or item["name"] in region:
                return item["code"], None

        # 2. 각 시/도의 시군구 목록에서 찾기
        for area_item in area_items:
            area_code = area_item["code"]
            sigungu_items = fetch_area_items(area_code=area_code)
            for sigungu in sigungu_items:
                if region in sigungu["name"] or sigungu["name"] in region:
                    return area_code, sigungu["code"]
                
    except Exception as e:
        print("❌ 지역 매핑 실패:", e)
    return None, None

def fetch_area_based_places(area_code: int, service_key: str, sigungu_code: Optional[int] = None, limit: int = 5) -> List[Dict]:
    """지역 기반 관광지 검색"""
    url = (
        f"https://apis.data.go.kr/B551011/KorPetTourService/areaBasedList"
        f"?serviceKey={service_key}&MobileOS=ETC&MobileApp=TestApp&_type=json"
        f"&pageNo=1&numOfRows={limit}&arrange=C&contentTypeId=12&areaCode={area_code}&listYN=Y"
    )
    if sigungu_code:
        url += f"&sigunguCode={sigungu_code}"
    try:
        result = subprocess.run(['curl', '-s', url], capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    except Exception as e:
        print("❌ 관광지 목록 조회 실패:", e)
    return []

def get_pet_tour_detail(contentid: int, service_key: str) -> str:
    url = (
        f"https://apis.data.go.kr/B551011/KorPetTourService/detailPetTour"
        f"?serviceKey={service_key}&MobileOS=ETC&MobileApp=TestApp&_type=json&contentId={contentid}"
    )
    try:
        result = subprocess.run(['curl', '-s', url], capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        items = data.get("response", {}).get("body", {}).get("items", {}).get("item", {})
        if isinstance(items, list):
            return items[0].get("acmpyPsblCpam", "정보 없음")
        elif isinstance(items, dict):
            return items.get("acmpyPsblCpam", "정보 없음")
    except Exception as e:
        print(f"❌ 상세 정보 조회 실패(contentid={contentid}):", e)
    return "정보 없음"

def fetch_pet_friendly_places_only(user_input: Dict, limit: int = 5) -> List[Dict]:
    region = user_input["region"]
    print(region)
    area_code, sigungu_code = match_region_to_codes(region)
    if not area_code:
        return []
    api_results = fetch_area_based_places(area_code, service_key, sigungu_code=sigungu_code, limit=limit)
    for place in api_results:
        place["pet_info"] = get_pet_tour_detail(place["contentid"], service_key)

    return api_results


# if __name__ == '__main__':
#     user_input = {
#         "region": "부산",
#         "pet_type": "강아지",
#         "days": 2,
#         "transport_mode": "자가용"
#     }
    
#     results = fetch_pet_friendly_places_only(user_input)

#     for i, item in enumerate(results, 1):
#         print(f"{i}. {item['title']} - {item.get('addr1', '주소 없음')}")
#         print(f"   🐶 동반 가능: {item['pet_info']}")