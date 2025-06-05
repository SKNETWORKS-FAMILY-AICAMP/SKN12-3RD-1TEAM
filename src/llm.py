""" Pet Travel Chatbot System"""
import logging
import traceback
from typing import Dict, List, Optional, Any
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from datetime import datetime
from dotenv import load_dotenv

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import vector_manger as vm
from module import get_category, get_user_parser, get_naver_map_link
from weather import get_weather, get_current_time

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 후보 장소 개수 설정
candiate_num = None

class Chatbot: # 챗봇 클래스
    """ Pet Travel Chatbot"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0.3)
        logger.info(" Chatbot initialized")
    
    def process_query(self, query: str, stream: bool = False) -> str:
        """Main processing pipeline"""
        try:
            # Check greetings first
            greeting_response = self.check_greeting(query)
            if greeting_response:
                return greeting_response
            
            # Analyze query
            categories = get_category(query)
            user_parsed = get_user_parser(query)
            
            # Search VectorDB
            results = vm.multiretrieve_by_category(query=query, categories=categories, k_each=10, top_k=10)
            
            # Handle weather separately
            if "날씨" in categories:
                region = user_parsed.get("region")
                # If the travel parser didn't find a region, try weather-specific parsing
                if not region or region == "" or region == "null":
                    region = self._extract_weather_region(query)
                
                if region:
                    results["날씨"] = self._get_weather_info(region)
                else:
                    results["날씨"] = [Document(page_content="지역을 명시해주세요. (예: 서울 날씨, 부산 날씨)", metadata={})]
            
            # Generate response
            return self._generate_response(query, user_parsed, results)
            
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return "죄송합니다. 요청을 처리하는 중 오류가 발생했습니다."
    
    def check_greeting(self, query: str) -> Optional[str]:
        """Check for greetings"""
        greetings = ["안녕", "안녕하세요", "hello", "hi"]
        query_lower = query.lower().strip()
        
        if any(greeting in query_lower for greeting in greetings):
            return """안녕하세요! 🐶 반려동물 여행 전문 도우미입니다.
                        
                        다음과 같은 도움을 드릴 수 있어요:
                        • 🗺️ 반려동물 동반 가능한 관광지 추천
                        • 🏨 펜션, 호텔 등 숙박시설 안내  
                        • 🚌 대중교통 이용 규정 안내
                        • ☀️ 여행지 날씨 정보
                        • 🐾 반려동물 동반 가능한 미용실, 동물병원, 애견용품점, 쇼핑물 등 추천
                        
                        예시: "제주도에 강아지랑 2박 3일 여행 가고 싶어"

                        어떤 여행을 계획하고 계신가요? 😊"""
        return None
    
    def _get_weather_info(self, region: str) -> List[Document]:
        """Get weather information"""
        try:
            weather_data = get_weather(region)
            current_time = get_current_time()
            
            if "error" not in weather_data:
                content = f"""현재 시간: {current_time['full_datetime']}
                                            도시: {weather_data['city']}
                                            기온: {weather_data['temperature']}°C
                                            습도: {weather_data['humidity']}%
                                            강수형태: {weather_data['precipitation_type']}
                                            풍속: {weather_data['wind_speed']} m/s"""
                return [Document(page_content=content, metadata=weather_data)]
            else:
                return [Document(page_content=f"날씨 정보 조회 실패: {weather_data['error']}", metadata={})]
        except Exception as e:
            logger.error(f"Weather error: {str(e)}")
            return [Document(page_content="날씨 정보를 가져오는데 실패했습니다.", metadata={})]
    
    def _analyze_query_categories(self, query: str) -> List[str]:
        """Analyze query and detect relevant categories"""
        try:
            return get_category(query)
        except Exception as e:
            logger.error(f"Error analyzing categories: {str(e)}")
            return ["관광지"]
    
    def _search_vector_db(self, query: str, categories: List[str]) -> Dict[str, List[Document]]:
        """Search vector database for each category"""
        try:
            return vm.multiretrieve_by_category(query=query, categories=categories, k_each=10, top_k=10)
        except Exception as e:
            logger.error(f"Error searching vector DB: {str(e)}")
            return {}
    
    def _generate_response(self, query: str, user_parsed: Dict[str, Any], 
                        results: Dict[str, List[Document]]) -> str:
        """Generate final response"""
        content_sections = []
        
        for category, docs in results.items():
            if not docs:
                continue
                
            category_content = f"### {category} 정보\n"
            
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                place_name = metadata.get("title", f"장소 {i}")
                map_link = get_naver_map_link(place_name) if place_name != f"장소 {i}" else "#"
                
                place_info = f"**{i}. [{place_name}]({map_link})**\n"
                place_info += doc.page_content + "\n\n"
                category_content += place_info
            
            content_sections.append(category_content)
        
        content = "\n".join(content_sections)
        
        template ="""
                당신은 반려동물과의 여행을 도와주는 감성적인 여행 플래너, 가이드 입니다.  
                아래 정보에 따라 **정확히 날씨 응답인지, 여행 코스 요청인지 구분하여 답변**하세요.
                
                ---
                🧾 사용자 질문: {query}  
                📍 지역: {region}  
                🐕 반려동물: {pet_type}  
                🗓️ 여행 기간: {days}일  
                🔍 제공된 정보:  
                {content}
                ---

                🎯 작성 지침:

                1. **사용자가 '날씨'만 요청한 경우에는**,  
                    - 해당지역 기온/날씨/풍속/습도 + 반려동물 외출시 유의사항 포함하여 작성해주세요
                    - 날씨 이외에는 정보를 작성하지 마세요 
                    - 외출 시 주의사항은 아래의 예시를 참고하여 애완동물과 함께 외출 시 주의사항을 작성해주세요.
                    [예시]
                    안녕하세요! 😊  
                    서울의 현재 날씨를 알려드릴게요.

                    🌤️ **오늘의 서울 날씨**  
                    - 🌡️ 기온: **18.5°C**  
                    - 💧 습도: **55%**  
                    - 🌬️ 바람: **1.5 m/s**  
                    - 🌤️ 날씨 상태: **맑음**

                    맑고 산뜻한 날씨네요!  
                    반려동물과 외출하시기 좋은 날이에요. 🐶💕

                    **🐾 외출 시 주의사항**  
                    - 햇빛이 강할 수 있으니 **그늘에서 쉬는 시간**을 자주 주세요.  
                    - **수분 보충**을 위해 물을 꼭 챙겨주세요.  
                    - **뜨거운 아스팔트**로부터 발바닥을 보호해 주세요.

                    오늘도 반려동물과 함께 행복한 하루 보내세요! 🌈✨
                    
                    
                2. 반대로, **'여행 코스' 요청일 경우에는** `🐾 1일차, 2일차` 등으로 일정을 구성하세요.
                    - 일정 구성: 오전 → 점심 → 오후 → 저녁
                    - 각 장소는 이름 + 설명 + 반려동물 동반 여부 
                    
                
                3. **날씨 + 여행**이 모두 포함된 질문이라면,  
                    👉 먼저 날씨 정보를 출력하고 → 아래에 여행 일정을 이어서 작성하세요.
                
                
                4. 숙소 추천은 마지막 또는 별도 섹션에 `🏨 숙소 추천` 제목으로 정리해주세요.
                    - 숙소명, 위치, 반려동물 동반 여부, 특징, 추가요금 여부
                
                5. 전체 말투는 따뜻하고 친근하게. 여행을 함께 준비하는 친구처럼 작성해주세요.
                
                6. 🐾, 🌳, 🍽️, 🐶, ✨ 등의 이모지를 적절히 활용해 가독성과 감성을 살려주세요.
                7. 마지막에는 감성적인 인사로 마무리해주세요.
                    - 예: “반려견과 함께하는 이번 여행이 오래도록 기억에 남기를 바랍니다! 🐕💕”
                """
        
        prompt = PromptTemplate.from_template(template)
        inputs = {
            "query": query,
            "region": user_parsed.get("region", "정보 없음"),
            "pet_type": user_parsed.get("pet_type", "정보 없음"),
            "days": user_parsed.get("days", "정보 없음"),
            "content": content or "관련 정보를 찾을 수 없습니다."
        }
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(inputs)

    def _extract_weather_region(self, query: str) -> Optional[str]:
        """Extract region from weather queries using simple parsing"""
        import re
        
        # Common weather keywords and modifiers to exclude
        weather_keywords = ["날씨", "기온", "온도", "비", "눈", "바람", "습도", "맑", "흐림", "현재", "지금", "오늘", "내일", "어때"]
        
        # Check if it's a weather query
        if not any(keyword in query for keyword in ["날씨", "기온", "온도"]):
            return None
            
        # Simple region extraction patterns - prioritize patterns that come before weather keywords
        city_patterns = [
            r'([가-힣]+(?:시|구|군|도))\s*(?:의\s*)?(?:날씨|기온|온도|현재)',  # 서울시 날씨, 강남구 날씨
            r'([가-힣]+)\s*(?:의\s*)?(?:날씨|기온|온도)',      # 서울 날씨, 서울의 날씨
            r'([가-힣]+)\s+(?:현재|지금)',                    # 서울 현재, 부산 지금
        ]
        
        for pattern in city_patterns:
            match = re.search(pattern, query)
            if match:
                candidate = match.group(1)
                if candidate not in weather_keywords:
                    return candidate
        
        # Fallback: find the first meaningful Korean word that's not a weather keyword
        words = re.findall(r'[가-힣]+', query)
        for word in words:
            if word not in weather_keywords and len(word) >= 2:
                return word
                
        return None

# Global instance
chatbot = None

def get_chatbot() -> Chatbot:
    """Get global  chatbot instance"""
    global chatbot
    if chatbot is None:
        chatbot = Chatbot()
    return chatbot


def process_query(query: str, stream: bool = False) -> str:
    """Process query using  chatbot"""
    chatbot = get_chatbot()
    return chatbot.process_query(query, stream)


def check_greeting(query: str) -> Optional[str]:
    """Check for greetings"""
    chatbot = get_chatbot()
    return chatbot.check_greeting(query)
