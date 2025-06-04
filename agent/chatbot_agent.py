import sys
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import json 
import openai
from datetime import datetime
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import Tool
import re
from langchain.docstore.document import Document
import logging
import traceback
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from typing import List

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import vector_manger as vm
from module import get_user_parser, get_category
from weather import get_weather, get_current_time
from fetch_pt_places import fetch_pet_friendly_places_only
from category_validator import CategoryValidator
from naver_map_utils import NaverMapUtils

# 로그 디렉토리 생성
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log')
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'chatbot.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY is required")

# global variables
threshold = 5

def safe_fetch_pet_friendly_places(user_parsed, n):
    """
    API 호출을 안전하게 감싸고 항상 Document 객체 리스트로 반환.
    실패 시 사용자에게 안내 메시지 포함.
    최대 3번까지 재시도하여 유효한 장소를 찾습니다.
    """
    MAX_RETRIES = 3
    
    # Input validation - check n first
    if not isinstance(n, int) or n <= 0:
        error_msg = "Invalid number of results requested"
        logger.error(error_msg)
        return [Document(page_content="요청된 결과 수가 올바르지 않습니다.", metadata={})]
        
    # Then check user_parsed
    if not user_parsed or not isinstance(user_parsed, dict):
        error_msg = "Invalid user_parsed parameter"
        logger.error(error_msg)
        return [Document(page_content="입력 형식이 올바르지 않습니다.", metadata={})]
    
    try:
        logger.info(f"Fetching pet friendly places for query: {user_parsed}")
        results = fetch_pet_friendly_places_only(user_parsed, n)
        docs = []
        
        for place in results or []:
            title = place.get('title', '이름 없음')
            logger.info(f"Processing place: {title}")
            
            # Validate place with Naver Maps
            if NaverMapUtils.is_valid_place(title):
                # Convert API result to Document
                content = title
                if place.get('addr1'):
                    content += f" - {place['addr1']}"
                if place.get('pet_info'):
                    content += f" (반려동물: {place['pet_info']})"
                    
                metadata = {
                    "title": title,
                    "addr1": place.get('addr1', ''),
                    "pet_info": place.get('pet_info', ''),
                    "type": place.get('type', '')
                }
                
                docs.append(Document(page_content=content, metadata=metadata))
            else:
                logger.warning(f"Invalid place filtered out: {title}")
        
        if not docs:
            logger.warning("No valid places found in API results")
            docs.append(Document(page_content="현재 유효한 장소를 찾을 수 없습니다.", metadata={}))
            
        return docs
        
    except Exception as e:
        logger.error(f"API call failed: {str(e)}\n{traceback.format_exc()}")
        return [Document(page_content="외부 데이터 호출에 실패했습니다. 네트워크 상태를 확인하거나 잠시 후 다시 시도해주세요.", metadata={})]

def generate_transportation_rules(pet_type: str = None, transport_type: str = None) -> str:
    """LLM을 사용하여 반려동물 대중교통 이용 규정 생성"""
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=openai_api_key
        )

        # 교통수단 키워드 매핑
        transport_keywords = {
            "기차": ["기차", "ktx", "열차", "철도", "korail", "코레일"],
            "버스": ["버스", "시내버스", "시외버스", "고속버스"],
            "지하철": ["지하철", "전철", "metro"],
            "택시": ["택시", "call", "콜택시"]
        }

        # 쿼리에서 교통수단 타입 추출
        if not transport_type:
            for t_type, keywords in transport_keywords.items():
                if any(keyword in query.lower() for keyword in keywords):
                    transport_type = t_type
                    break

        transport_template = PromptTemplate.from_template("""
        다음 정보를 바탕으로 반려동물 대중교통 이용 규정을 안내해주세요:
        반려동물 종류: {pet_type}
        교통수단: {transport_type}

        규칙:
        1. 친근하고 명확한 말투 사용
        2. 이모지 적절히 활용
        3. 해당 교통수단의 구체적인 규정만 설명
        4. 반려동물 동반 시 주의사항 포함
        5. 필요한 준비물 안내

        {format_guide}
        """)

        # 교통수단별 응답 포맷 가이드
        format_guides = {
            "기차": """
            응답 형식:
            ### 🚄 기차 이용 안내
            
            #### KTX/일반열차 이용 규정
            [구체적인 규정]
            
            #### 이용 시 주의사항
            [주의사항 목록]
            
            #### 준비물 안내
            [필요한 준비물 목록]
            """,
            "버스": """
            응답 형식:
            ### 🚌 버스 이용 안내
            
            #### 버스 이용 규정
            [구체적인 규정]
            
            #### 이용 시 주의사항
            [주의사항 목록]
            
            #### 준비물 안내
            [필요한 준비물 목록]
            """,
            "지하철": """
            응답 형식:
            ### 🚇 지하철 이용 안내
            
            #### 지하철 이용 규정
            [구체적인 규정]
            
            #### 이용 시 주의사항
            [주의사항 목록]
            
            #### 준비물 안내
            [필요한 준비물 목록]
            """,
            "택시": """
            응답 형식:
            ### 🚕 택시 이용 안내
            
            #### 택시 이용 규정
            [구체적인 규정]
            
            #### 이용 시 주의사항
            [주의사항 목록]
            
            #### 준비물 안내
            [필요한 준비물 목록]
            """
        }

        # 특정 교통수단이 지정된 경우 해당 포맷만 사용
        format_guide = format_guides.get(transport_type, """
        응답 형식:
        ### 🚌 대중교통 이용 안내
        
        #### 이용 가능한 교통수단
        [교통수단별 규정]
        
        #### 이용 시 주의사항
        [주의사항 목록]
        
        #### 준비물 안내
        [필요한 준비물 목록]
        """)

        chain = transport_template | llm | StrOutputParser()
        rules = chain.invoke({
            "pet_type": pet_type or "반려동물",
            "transport_type": transport_type or "대중교통",
            "format_guide": format_guide
        })
        
        return rules.strip()
        
    except Exception as e:
        logger.error(f"Error generating transportation rules: {str(e)}")
        return """### 🚌 대중교통 이용 안내
        
죄송합니다. 현재 대중교통 규정 정보를 조회할 수 없습니다.
다시 시도해 주시거나 각 교통수단 운영기관에 직접 문의해주세요."""

def generate_travel_course(tourist_spots: List[Document], days: int = None, city: str = None) -> str:
    """LLM을 사용하여 여행 코스 생성"""
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=openai_api_key
        )

        # 관광지 정보를 문자열로 변환
        spots_info = []
        for spot in tourist_spots:
            title = spot.metadata.get("title", "")
            addr = spot.metadata.get("addr1", "정보 없음")
            pet_info = spot.metadata.get("pet_info", "정보 없음")
            spots_info.append(f"{title} (주소: {addr}, 반려동물: {pet_info})")

        course_template = PromptTemplate.from_template("""
        다음 정보를 바탕으로 여행 코스를 생성해주세요:
        도시: {city}
        여행 일수: {days}일
        방문 가능한 관광지 목록:
        {spots}

        규칙:
        1. 각 일자별로 적절한 수의 관광지 배치
        2. 이동 동선을 고려한 효율적인 코스 구성
        3. 체크인/아웃 시간을 고려한 일정 배치
        4. 반려동물 동반 특성을 고려한 코스 구성
        5. 각 장소별 추천 포인트와 소요시간 포함
        6. 관광지가 부족한 경우 주변 추천 활동 포함
        7. 하루 최대 3-4곳 방문 권장

        Few-shot 예시:

        입력 예시 1:
        도시: 제주도
        여행 일수: 3일
        관광지: 협재해수욕장, 한라산, 성산일출봉, 카페거리, 올레길

        출력 예시 1:
        ### 🎯 추천 여행 코스
        여행의 즐거움을 더할 수 있는 코스를 추천해드립니다!

        #### 🌅 1일차
        첫째 날은 도착 후 여유로운 일정으로 시작해보세요!
        
        ##### 협재해수욕장
        - 💡 추천 포인트: 반려동물과 함께 해변 산책을 즐기기 좋은 오후 시간대 방문 추천
        - ⏰ 추천 소요시간: 1-2시간
        - 🔍 주변 추천: 근처 반려동물 동반 카페에서 휴식
        
        ##### 카페거리
        - 💡 추천 포인트: 반려동물 동반 가능한 카페에서 여유로운 휴식
        - ⏰ 추천 소요시간: 1-2시간

        #### 🌅 2일차        
        ##### 성산일출봉
        - 💡 추천 포인트: 아침 일출 명소, 반려동물과 함께하는 트레킹
        - ⏰ 추천 소요시간: 2-3시간
        
        ##### 올레길
        - 💡 추천 포인트: 반려동물과 함께하는 여유로운 올레길 산책
        - ⏰ 추천 소요시간: 2-3시간
        - 🔍 주변 추천: 올레길 주변 포토스팟에서 인생샷

        #### 🌅 3일차
        마지막 날은 체크아웃 시간을 고려한 일정입니다.
        
        ##### 한라산
        - 💡 추천 포인트: 오전에 방문하여 날씨 좋을 때 산책로 코스 추천
        - ⏰ 추천 소요시간: 2-3시간
        - 🔍 주변 추천: 근처 전망 좋은 카페에서 마무리

        입력 예시 2:
        도시: 경주
        여행 일수: 2일
        관광지: 신라왕경숲

        출력 예시 2:
        ### 🎯 추천 여행 코스
        여행의 즐거움을 더할 수 있는 코스를 추천해드립니다!

        #### 🌅 1일차
        첫째 날은 도착 후 여유로운 일정으로 시작해보세요!
        
        ##### 신라왕경숲 (오후)
        - 💡 추천 포인트: 넓은 숲속에서 반려동물과 함께 산책하기 좋은 장소
        - ⏰ 추천 소요시간: 2-3시간
        - 🔍 주변 추천: 
          * 근처 반려동물 동반 카페에서 휴식
          * 황리단길 산책
          * 동궁과 월지(안압지) 야경 감상

        #### 🌅 2일차
        마지막 날은 체크아웃 시간을 고려한 일정입니다.
        
        ##### 신라왕경숲 (아침)
        - 💡 추천 포인트: 아침 산책으로 상쾌한 하루 시작
        - ⏰ 추천 소요시간: 1-2시간
        - 🔍 주변 추천:
          * 첨성대 주변 산책
          * 대릉원 돌담길 산책
          * 카페거리에서 브런치

        입력 예시 3:
        도시: 속초
        여행 일수: 1일
        관광지: 외옹치해변, 대포항

        출력 예시 3:
        ### 🎯 추천 여행 코스
        여행의 즐거움을 더할 수 있는 코스를 추천해드립니다!

        #### 🌅 당일 코스
        ##### 외옹치해변 (오전)
        - 💡 추천 포인트: 아침 바다 산책과 일출 감상
        - ⏰ 추천 소요시간: 1-2시간
        - 🔍 주변 추천: 해변 카페에서 모닝커피
        
        ##### 대포항 (점심~오후)
        - 💡 추천 포인트: 반려동물과 함께하는 항구 산책
        - ⏰ 추천 소요시간: 2-3시간
        - 🔍 주변 추천:
          * 대포항 수산시장 구경
          * 반려동물 동반 가능한 횟집
          * 등대 전망대 산책

        실제 응답을 위한 형식:
        ### 🎯 추천 여행 코스
        여행의 즐거움을 더할 수 있는 코스를 추천해드립니다!

        [일자별 코스 구성]
        """)

        chain = course_template | llm | StrOutputParser()
        course = chain.invoke({
            "city": city or "여행지",
            "days": days or 1,
            "spots": "\n".join(spots_info)
        })
        
        return course.strip()
        
    except Exception as e:
        logger.error(f"Error generating travel course: {str(e)}")
        return """### 🎯 추천 여행 코스
        
죄송합니다. 현재 여행 코스 생성에 문제가 발생했습니다.
다시 시도해 주시거나 다른 방식으로 안내해 드리겠습니다."""

def generate_category_content(results, user_categories, city, query: str = ""):
    """카테고리별 컨텐츠 생성"""
    logger.info(f"Generating content for categories: {user_categories} in city: {city}")
    content_sections = []
    
    # 인사말 및 요약 생성
    greeting = generate_greeting(city, user_categories)
    content_sections.append(f"### {greeting}\n")
    
    # 대중교통 정보 처리
    if "대중교통" in user_categories:
        # 쿼리에서 반려동물 종류 추출 (예: "리트리버")
        pet_type = None
        for keyword in ["리트리버", "푸들", "말티즈", "치와와", "포메라니안"]:  # 필요한 견종 추가
            if keyword.lower() in query.lower():
                pet_type = keyword
                break
        transport_rules = generate_transportation_rules(pet_type)
        content_sections.append(transport_rules)
    
    # 날씨 정보 처리
    if "날씨" in user_categories and city:
        try:
            weather = get_weather(city)
            if "error" not in weather:
                weather_info = (
                    "### 날씨 정보\n"
                    f"- 도시: {weather['city']}\n"
                    f"- 기온: {weather['temperature']}°C\n"
                    f"- 습도: {weather['humidity']}%\n"
                    f"- 강수형태: {weather['precipitation_type']}\n"
                )
                content_sections.append(weather_info)
            else:
                content_sections.append("### 날씨 정보\n현재 날씨 정보를 조회할 수 없습니다.\n")
        except Exception as e:
            logger.error(f"Weather fetch failed: {str(e)}")
            content_sections.append("### 날씨 정보\n현재 날씨 정보를 조회할 수 없습니다.\n")
    
    # 여행 코스 추천 섹션 추가
    if "관광지" in user_categories:
        tourist_spots = []
        if "관광지" in results:
            for place in results["관광지"]:
                if isinstance(place, Document):
                    tourist_spots.append(place)
        
        if tourist_spots:
            # 여행 일수 확인
            days = None
            try:
                days_match = re.search(r'(\d+)박', query)
                if days_match:
                    days = int(days_match.group(1)) + 1  # N박의 경우 N+1일
            except:
                days = None
            
            # LLM을 사용하여 여행 코스 생성
            course_content = generate_travel_course(tourist_spots, days, city)
            content_sections.append(course_content)
            
            # 여행 팁 추가
            content_sections.append("\n#### ✨ 여행 팁")
            content_sections.append("- 🕐 각 관광지마다 추천 소요시간을 참고하여 여유있게 일정을 잡으세요.")
            content_sections.append("- 🐕 반려동물과 함께할 때는 목줄과 배변봉투를 필수로 준비해주세요.")
            content_sections.append("- 📸 인생샷 스팟이 많으니 카메라 준비 필수!")
            content_sections.append("- 🚗 주차 공간이 있는지 미리 확인하시면 좋습니다.")
            if days and days > 1:
                content_sections.append("- 🏨 체크인/체크아웃 시간을 고려하여 일정을 조율하세요.")
                content_sections.append("- 🌦️ 날씨를 미리 확인하고 일정을 유동적으로 조정하세요.")
    
    # 숙박 정보 처리
    if "숙박" in user_categories:
        content_sections.append("\n### 🏨 숙박 정보")
        if "숙박" in results and results["숙박"]:
            for place in results["숙박"]:
                if isinstance(place, Document):
                    title = place.metadata.get("title", "")
                    addr = place.metadata.get("addr1", "정보 없음")
                    pet_info = place.metadata.get("pet_info", "정보 없음")
                    map_link = NaverMapUtils.get_map_link(title)
                    
                    accommodation_info = [
                        f"#### {title}",
                        f"- 📍 [네이버 지도]({map_link})",
                        f"- 🏠 주소: {addr}",
                        f"- 🐕 반려동물: {pet_info}",
                        f"- 💡 숙소 특징: {get_accommodation_highlight(title)}\n"
                    ]
                    content_sections.append("\n".join(accommodation_info))
            
            # 숙박 팁 추가
            content_sections.append("\n#### ✨ 숙박 팁")
            content_sections.append("- 🐕 반려동물 동반 시 미리 예약하시는 것을 추천드립니다.")
            content_sections.append("- 📞 체크인 시간을 미리 확인하세요.")
            content_sections.append("- 🧹 반려동물 용품(방석, 배변패드 등)을 준비하면 좋습니다.")
        else:
            content_sections.append("현재 이용 가능한 숙박 시설 정보가 없습니다.")
    
    return "\n".join(content_sections)

def get_spot_highlight(spot_name: str) -> str:
    """관광지별 하이라이트 정보 반환"""
    highlights = {
        "외옹치해변": "아름다운 일출 명소이며, 반려동물과 함께 산책하기 좋은 해변입니다. 특히 아침 시간대 방문을 추천합니다.",
        "대포항": "신선한 해산물을 즐길 수 있는 곳으로, 반려동물과 함께 여유로운 산책이 가능합니다. 일몰 시간대가 특히 아름답습니다.",
        "영금정": "속초 시내가 한눈에 보이는 전망 명소로, 반려동물과 함께 인생샷을 남기기 좋은 장소입니다."
    }
    return highlights.get(spot_name, "아름다운 경관과 함께 반려동물과 특별한 추억을 만들 수 있는 장소입니다.")

def get_accommodation_highlight(accommodation_name: str) -> str:
    """숙소별 하이라이트 정보 반환"""
    highlights = {
        "설악금호리조트": "반려동물 전용 용품이 구비되어 있으며, 주변 산책로가 잘 조성되어 있습니다. 8kg 이하 소형견 동반 가능합니다."
    }
    return highlights.get(accommodation_name, "반려동물과 함께 편안한 휴식을 취할 수 있는 숙소입니다.")

def create_dynamic_prompt(query: str, user_categories: list):
    """사용자 질문 의도에 맞는 동적 프롬프트 생성"""
    
    # 카테고리별 응답 형식과 지시사항
    category_formats = {
        "관광명소": {
            "single": "관광지 정보와 추천 코스를 상세히 설명해주세요.",
            "combined": "관광지 추천과 방문 순서를 계획에 포함해주세요."
        },
        "숙박": {
            "single": "숙박 시설 정보와 예약 시 주의사항을 상세히 설명해주세요.",
            "combined": "숙박 시설 추천과 예약 관련 정보를 계획에 포함해주세요."
        },
        "대중교통": {
            "single": "대중교통 이용 방법과 규정을 상세히 설명해주세요.",
            "combined": "대중교통 이용 방법과 주의사항을 계획에 포함해주세요."
        },
        "날씨": {
            "single": "현재 날씨 상황과 여행 시 주의사항을 상세히 설명해주세요.",
            "combined": "날씨에 따른 여행 팁과 준비물을 계획에 포함해주세요."
        }
    }
    
    # 단일 카테고리 응답인지 확인
    is_single_category = len(user_categories) == 1
    
    instructions = []
    format_type = "single" if is_single_category else "combined"
    
    # 카테고리별 지시사항 추가
    for category in user_categories:
        if category in category_formats:
            instructions.append(category_formats[category][format_type])
    
    # 기본 응답 형식 지정
    response_format = [
        "",
        "응답 형식:",
        "1. 마크다운 형식으로 작성",
        "2. 각 카테고리를 명확히 구분",
        "3. 구체적인 정보와 근거 포함",
        "4. 실제 장소명과 데이터 활용"
    ]
    
    # 최종 프롬프트 조합
    prompt_parts = [
        f"다음 질문에 답변해주세요: {query}",
        "제공된 정보를 바탕으로 다음 사항들을 포함하여 응답해주세요:",
        *instructions,
        *response_format
    ]
    
    return "\n".join(prompt_parts)

def process_query(query: str, stream: bool = True):
    """사용자 쿼리 처리 파이프라인 (API/DB 안정성 강화)"""
    logger.info(f"Processing query: {query}")
    
    try:
        # 1. 사용자 입력 파싱
        user_parsed = get_user_parser(query)
        if not user_parsed:
            raise ValueError("Failed to parse user query")
            
        user_categories = get_category(query)
        if not user_categories:
            raise ValueError("Failed to identify categories from query")
            
        city = user_parsed.get("region", "")
        logger.info(f"Parsed query - Categories: {user_categories}, City: {city}")
        
        # 대중교통 카테고리만 있는 경우 바로 LLM 응답
        if len(user_categories) == 1 and "대중교통" in user_categories:
            # 반려동물 종류 추출
            pet_type = None
            for keyword in ["리트리버", "푸들", "말티즈", "치와와", "포메라니안"]:
                if keyword.lower() in query.lower():
                    pet_type = keyword
                    break
            
            # 교통수단 타입 추출
            transport_type = None
            transport_keywords = {
                "기차": ["기차", "ktx", "열차", "철도", "korail", "코레일"],
                "버스": ["버스", "시내버스", "시외버스", "고속버스"],
                "지하철": ["지하철", "전철", "metro"],
                "택시": ["택시", "call", "콜택시"]
            }
            
            for t_type, keywords in transport_keywords.items():
                if any(keyword in query.lower() for keyword in keywords):
                    transport_type = t_type
                    break
            
            # 인사말 생성
            greeting = generate_greeting(city, user_categories)
            # 대중교통 규정 생성
            transport_rules = generate_transportation_rules(pet_type, transport_type)
            
            # 대중교통 카테고리만 있는 경우 바로 반환
            return f"### {greeting}\n\n{transport_rules}"

        # 2. 숙박/관광지 관련 검색일 경우에만 벡터 검색 및 API 호출
        combined_results = {"관광지": [], "숙박": []}
        if any(category in ["숙박", "관광지"] for category in user_categories):
            # 벡터 검색 결과 가져오기
            vector_results = vm.multiretrieve_by_category(query, user_categories)
            logger.info(f"Vector search results: {vector_results}")
            
            # API 결과 가져오기
            api_results = safe_fetch_pet_friendly_places(user_parsed, 5)
            logger.info(f"API results: {api_results}")
            
            # 결과 통합 및 검증
            for category in ["숙박", "관광지"]:
                if category in user_categories:
                    # 벡터 검색 결과 처리
                    if category in vector_results:
                        for doc in vector_results[category]:
                            title = doc.metadata.get("title", "")
                            if title and NaverMapUtils.is_valid_place(title):
                                combined_results[category].append(doc)
                    
                    # API 결과 처리
                    for place in api_results:
                        if isinstance(place, Document):
                            title = place.metadata.get("title", "").lower()
                            
                            # 숙박 시설 키워드
                            accommodation_keywords = [
                                "리조트", "호텔", "숙소", "펜션", "게스트하우스", 
                                "민박", "콘도", "모텔", "숙박", "스테이", "레지던스",
                                "하우스", "빌라", "룸", "스위트", "캠핑", "park",
                                "글램핑", "camping"
                            ]
                            
                            # 관광지 키워드
                            tourist_keywords = [
                                "해변", "항", "공원", "오름", "폭포", "계곡", 
                                "동굴", "산", "숲", "정", "농장", "박물관", 
                                "미술관", "성", "절", "유적", "전망대", "단지",
                                "거리", "마을", "릉", "궁", "길", "관광"
                            ]
                            
                            # 장소 분류
                            if category == "숙박" and any(keyword in title for keyword in accommodation_keywords):
                                place.metadata["type"] = "숙박"
                                if title not in [doc.metadata.get("title", "") for doc in combined_results["숙박"]]:
                                    combined_results["숙박"].append(place)
                            elif category == "관광지" and any(keyword in title for keyword in tourist_keywords):
                                place.metadata["type"] = "관광지"
                                if title not in [doc.metadata.get("title", "") for doc in combined_results["관광지"]]:
                                    combined_results["관광지"].append(place)
                            elif category == "관광지":  # 키워드에 없는 관광지도 포함
                                place.metadata["type"] = "관광지"
                                if title not in [doc.metadata.get("title", "") for doc in combined_results["관광지"]]:
                                    combined_results["관광지"].append(place)
        
        # 컨텐츠 생성
        content = generate_category_content(combined_results, user_categories, city, query)
        
        # 검증된 장소가 없는 경우 메시지 추가
        if any(category in ["숙박", "관광지"] for category in user_categories):
            total_places = sum(len(places) for category, places in combined_results.items())
            if total_places == 0:
                content = "### 안내\n검증된 장소를 찾을 수 없습니다.\n\n" + content
        
        return content
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return "죄송합니다. 요청을 처리하는 중에 오류가 발생했습니다. 다시 시도해 주세요."


# 일반적인 대화 내용 생성 
def check_greeting(query: str) -> str:
    """일반적인 대화 내용 생성"""
    greetings = ['안녕', 'hello', '안녕하세요', '반가워', '하이', 'hi']
    thanks = ['감사합니다', '고마워', '감사', '고마워요', '감사합니다', '고마워요']
    help_words = ["도움", "어떻게 써", "무슨 기능", "설명해줘"]
    query = query.lower()
    
    # if any(greeting in query for greeting in greetings):
        # return "안녕하세요! 저는 🐶 반려동물과 함께 여행 도 움봇입니다. 무엇을 도와드릴까요?"
    if any(word in query for word in greetings):
        return "안녕하세요! 여행 관련해서 궁금한 걸 물어보세요 😊" # 인사응답
    elif any(word in query for word in thanks):
        return "언제든지 도와드릴게요! 또 궁금한 거 있으신가요?" # 감사응답
    elif any(word in query for word in help_words):
        return "이 챗봇은 여행 코스 추천, 숙박 정보, 대중교통 규정 안내를 도와드려요. 예: '제주도 여행 추천해줘'" # 도움말 응답
    else:
        return  None # 이래야 대화가 계속 진행됨

def generate_greeting(city: str, categories: List[str]) -> str:
    """LLM을 사용하여 사용자 질문에 맞는 인사말 생성"""
    try:
        # LLM 초기화
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=openai_api_key
        )

        # 프롬프트 템플릿 생성
        greeting_template = PromptTemplate.from_template("""
        다음 정보를 바탕으로 여행 안내 인사말을 생성해주세요:
        도시: {city}
        카테고리: {categories}

        규칙:
        1. 친근하고 자연스러운 말투 사용
        2. 이모지 적절히 활용
        3. 한 문장으로 간단하게 작성
        4. 카테고리가 여러 개면 자연스럽게 나열
        5. "안내해드리겠습니다"로 끝나도록 작성

        예시:
        - 대구의 맛집과 관광 정보를 한 번에 알려드리겠습니다 🌟
        - 제주도의 숙박, 관광 및 날씨 정보를 상세히 안내해드리겠습니다 ✨
        - 부산의 반려동물 동반 가능한 해수욕장 정보를 안내해드리겠습니다 🏖

        응답:
        """)

        # 카테고리 문자열 생성
        categories_str = json.dumps(categories, ensure_ascii=False) if categories else "[]"
        
        # LLM 체인 생성 및 실행
        chain = greeting_template | llm | StrOutputParser()
        greeting = chain.invoke({
            "city": city or "전국",
            "categories": categories_str
        })
        
        return greeting.strip()
        
    except Exception as e:
        logger.error(f"Error generating greeting: {str(e)}")
        # 오류 발생 시 기본 인사말 반환
        if city:
            return f"{city} 여행 정보를 안내해드리겠습니다 🏖"
        return "여행 정보를 안내해드리겠습니다 🌟"



# 실행 예시
# if __name__ == "__main__":
#     logger.info("Starting test queries")
#     # 테스트 쿼리 예시들
#     test_queries = [
#         "속초로 여행갈려고하는데 버스 규정이 어떻게돼?",  # 대중교통만
#         "속초의 날씨 어때?",  # 날씨만
#         "속초 버스로 여행갈려고하는데 현재 날씨는?",  # 날씨 + 대중교통
#         "속초 숙박시설 추천해줘",  # 숙박만
#         "속초 관광지 추천"  # 관광만
#     ]
    
#     # 단위 쿼리 테스트 출력 
#     for query in test_queries:
#         try:
#             logger.info(f"Testing query: {query}")
#             result = process_query(query, stream=False)
#             logger.info(f"Query successful: {query}")
#             print(f"질문: {query}")
#             print(f"결과: {result}")
#             print('-'*50)
#         except Exception as e:
#             logger.error(f"Test query failed: {query}\n{str(e)}\n{traceback.format_exc()}")