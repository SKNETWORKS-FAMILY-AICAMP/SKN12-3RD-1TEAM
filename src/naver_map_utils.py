import os
import urllib.parse
import requests
import logging

# 향후 실제 데이터를 기반으로 장소를 선별할 예정으로 사전에 코드 작성 
logger = logging.getLogger(__name__)

class NaverMapUtils:
    # 알려진 장소 ID 매핑
    PLACE_IDS = {
        "속초해수욕장": "13994080",
        "속초관광수산시장": "11491456",
        "속초중앙시장": "13545523",
        "설악산국립공원": "11491297",
        # 다른 장소들의 ID도 추가 가능
    }
    
    @staticmethod
    def get_map_link(place_name: str) -> str:
        """
        네이버 지도 링크 생성
        장소 ID가 있는 경우 직접 링크 사용, 없는 경우 검색 링크 사용
        
        Args:
            place_name (str): 장소 이름
            
        Returns:
            str: 네이버 지도 링크
        """
        if place_name in NaverMapUtils.PLACE_IDS:
            return f"https://map.naver.com/p/entry/place/{NaverMapUtils.PLACE_IDS[place_name]}"
        else:
            # 검색 URL로 폴백
            encoded_name = urllib.parse.quote(place_name)
            return f"https://map.naver.com/p/search/{encoded_name}"
    
    @staticmethod
    def is_valid_place(place_name: str) -> bool:
        """
        네이버 지도 API를 통해 장소 유효성 확인
        
        Args:
            place_name (str): 검증할 장소 이름
            
        Returns:
            bool: 유효한 장소면 True, 아니면 False
        """
        try:
            url = 'https://openapi.naver.com/v1/search/local.json'
            params = {
                'query': place_name,
                'display': 1  # 결과 개수 제한
            }
            headers = {
                'X-Naver-Client-Id': os.getenv('NAVER_CLIENT_KEY'),
                'X-Naver-Client-Secret': os.getenv('NAVER_CLIENT_SECRET_KEY')
            }
            
            response = requests.get(url, params=params, headers=headers)
            if response.status_code != 200:
                logger.warning(f"Naver API request failed: {response.status_code}")
                return False
                
            data = response.json()
            return len(data.get('items', [])) > 0
            
        except Exception as e:
            logger.error(f"Error validating place {place_name}: {str(e)}")
            return False 