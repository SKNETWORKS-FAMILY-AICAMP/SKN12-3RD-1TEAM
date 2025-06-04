from typing import Dict, List, Optional
import logging
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

# 카테고리별 장소 타입 정의
CATEGORY_TYPES = {
    "관광명소": ["관광지", "공원", "전망대", "해변", "박물관", "미술관"],
    "숙박": ["호텔", "펜션", "게스트하우스", "리조트", "모텔"],
    "식당": ["음식점", "카페", "레스토랑", "베이커리"],
    "대중교통": ["버스터미널", "기차역", "버스정류장"]
}

class CategoryValidator:
    @staticmethod
    def validate_place_type(category: str, place_type: str) -> bool:
        """
        주어진 장소 타입이 해당 카테고리에 유효한지 검증
        
        Args:
            category (str): 카테고리 이름
            place_type (str): 장소 타입
            
        Returns:
            bool: 유효한 타입이면 True, 아니면 False
        """
        if not place_type or not category:
            logger.warning(f"Invalid input - category: {category}, place_type: {place_type}")
            return False
            
        if category not in CATEGORY_TYPES:
            logger.warning(f"Unknown category: {category}")
            return False
            
        # 대소문자 구분 없이 타입 검사
        place_type_lower = place_type.lower()
        valid_types = [t.lower() for t in CATEGORY_TYPES[category]]
        
        is_valid = any(t in place_type_lower for t in valid_types)
        if not is_valid:
            logger.warning(f"Type mismatch - category: {category}, place_type: {place_type}")
            
        return is_valid

    @staticmethod
    def filter_places_by_category(category: str, places: List[Document]) -> List[Document]:
        """
        카테고리에 맞는 장소만 필터링
        
        Args:
            category (str): 카테고리 이름
            places (List[Document]): 장소 문서 리스트
            
        Returns:
            List[Document]: 필터링된 장소 리스트
        """
        filtered_places = []
        for place in places:
            if not isinstance(place, Document):
                logger.warning(f"Invalid place object: {place}")
                continue
                
            metadata = place.metadata
            if not metadata:
                logger.warning(f"Place has no metadata: {place}")
                continue
                
            place_type = metadata.get("type")
            if not place_type:
                logger.warning(f"Place has no type: {metadata.get('title', 'Unknown')}")
                continue
                
            if CategoryValidator.validate_place_type(category, place_type):
                filtered_places.append(place)
            else:
                logger.info(f"Filtered out place {metadata.get('title', 'Unknown')} with type {place_type} from category {category}")
                
        return filtered_places

    @staticmethod
    def get_place_info(place: Document, map_link_func) -> Optional[Dict[str, str]]:
        """
        장소 정보를 구조화된 형태로 반환
        
        Args:
            place (Document): 장소 문서
            map_link_func: 네이버 지도 링크 생성 함수
            
        Returns:
            Optional[Dict[str, str]]: 구조화된 장소 정보 또는 None
        """
        try:
            metadata = place.metadata
            if not metadata or "title" not in metadata:
                logger.warning(f"Invalid place metadata: {metadata}")
                return None
                
            title = metadata["title"]
            map_link = map_link_func(title)
            
            info = {
                "name": f"[{title}]({map_link})",
                "type": f"유형: {metadata.get('type', '정보 없음')}",
                "address": f"주소: {metadata.get('addr1', '정보 없음')}",
                "tel": f"연락처: {metadata.get('tel', '정보 없음')}",
                "pet_info": f"반려동물 동반: {metadata.get('pet_info', '정보 없음')}"
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error processing place info: {str(e)}")
            return None

    @staticmethod
    def format_place_info(info: Dict[str, str]) -> str:
        """
        장소 정보를 문자열로 포맷팅
        
        Args:
            info (Dict[str, str]): 구조화된 장소 정보
            
        Returns:
            str: 포맷팅된 장소 정보
        """
        if not info:
            return "장소 정보 없음"
            
        return " | ".join(info.values()) 