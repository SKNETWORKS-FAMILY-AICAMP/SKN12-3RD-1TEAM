import logging
import traceback
from typing import Dict, List, Optional, Tuple, Any
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Import existing modules
import vector_manger as vm
from module import get_category, get_user_parser, get_naver_map_link
from fetch_pt_places import fetch_pet_friendly_places_only
from weather import get_weather, get_current_time
from vectordb_updater import VectorDBUpdater

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:
    """
    Enhanced retrieval system with automatic category routing and dynamic VectorDB augmentation
    """
    
    def __init__(self, 
                min_results_threshold: int = 3,
                quality_threshold: float = 0.7,
                max_external_results: int = 30,  # 최대 외부 API fetch 개수 넉넉히
                enable_db_updates: bool = True):
        """
        Initialize the enhanced retriever
        
        Args:
            min_results_threshold: Minimum number of results required before external API call
            quality_threshold: Minimum similarity score for considering results as high quality
            max_external_results: Maximum number of results to fetch from external APIs
            enable_db_updates: Whether to save new data to VectorDB
        """
        self.min_results_threshold = min_results_threshold
        self.quality_threshold = quality_threshold
        self.max_external_results = max_external_results
        self.enable_db_updates = enable_db_updates
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0.3
        )
        
        # Initialize VectorDB updater if updates are enabled
        self.db_updater = VectorDBUpdater() if enable_db_updates else None
        
        # Category to external API mapping
        self.external_api_mapping = {
            "관광지": self._fetch_tourist_attractions,
            "숙박": self._fetch_accommodations,
            # "대중교통": self._fetch_transport_info,  # Can be added later
        }
    
    def get_total_needed_places(self, days: Optional[Any]) -> int:
        try:
            days = int(days)
            if days < 1:
                days = 1
        except (TypeError, ValueError):
            days = 1
        return days * 4

    def process_query(self, query: str, stream: bool = False) -> str:
        """
        Main processing pipeline for user queries
        
        Args:
            query: User input query
            stream: Whether to stream the response
            
        Returns:
            Generated response or stream generator
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Step 1: Query Analysis
            categories = self._analyze_query_categories(query)
            user_parsed = self._parse_user_info(query)
            
            logger.info(f"Detected categories: {categories}")
            logger.info(f"Parsed user info: {user_parsed}")
            
            # Step 1.5: Determine needed places by days
            days = user_parsed.get("days", 1)
            total_needed = self.get_total_needed_places(days)
            
            # Step 2: Initial VectorDB Search (동적으로 개수 조정)
            initial_results = self._search_vector_db(query, categories, k_each=total_needed, top_k=total_needed)
            
            # Step 3: Result Quality Assessment
            quality_assessment = self._assess_result_quality(initial_results, categories, total_needed)
            
            # Step 4: Dynamic Augmentation if needed (total_needed 전달)
            final_results = self._augment_results_if_needed(
                query, user_parsed, categories, initial_results, quality_assessment, total_needed
            )
            
            # Step 5: Generate Final Response
            return self._generate_response(query, user_parsed, final_results, stream)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}\n{traceback.format_exc()}")
            return "죄송합니다. 요청을 처리하는 중 오류가 발생했습니다. 다시 시도해 주세요."
    
    def _analyze_query_categories(self, query: str) -> List[str]:
        """Analyze query and detect relevant categories"""
        try:
            return get_category(query)
        except Exception as e:
            logger.error(f"Error analyzing categories: {str(e)}")
            return ["관광지"]  # Default fallback
    
    def _parse_user_info(self, query: str) -> Dict[str, Any]:
        """Parse user information from query"""
        try:
            return get_user_parser(query)
        except Exception as e:
            logger.error(f"Error parsing user info: {str(e)}")
            return {"region": None, "pet_type": None, "days": None}
    
    def _search_vector_db(self, query: str, categories: List[str], k_each: int = 5, top_k: int = 5) -> Dict[str, List[Document]]:
        """Search vector database for each category"""
        try:
            return vm.multiretrieve_by_category(
                query=query,
                categories=categories,
                k_each=k_each,
                top_k=top_k
            )
        except Exception as e:
            logger.error(f"Error searching vector DB: {str(e)}")
            return {}
    
    def _assess_result_quality(self, results: Dict[str, List[Document]], 
                                categories: List[str], total_needed: int) -> Dict[str, Dict[str, Any]]:
        """
        Assess the quality and sufficiency of search results
        
        Returns:
            Dict with quality assessment for each category
        """
        assessment = {}
        
        for category in categories:
            if category == "날씨":
                continue  # Weather is handled separately
                
            category_results = results.get(category, [])
            result_count = len(category_results)
            
            # Assess quantity
            sufficient_quantity = result_count >= total_needed
            
            # Assess quality (simple heuristic - can be improved)
            avg_content_length = 0
            if category_results:
                avg_content_length = sum(len(doc.page_content) for doc in category_results) / len(category_results)
            
            sufficient_quality = avg_content_length > 50  # Minimum content length
            
            assessment[category] = {
                "result_count": result_count,
                "sufficient_quantity": sufficient_quantity,
                "sufficient_quality": sufficient_quality,
                "needs_augmentation": not (sufficient_quantity and sufficient_quality)
            }
            
            logger.info(f"Quality assessment for {category}: {assessment[category]}")
        
        return assessment
    
    def _augment_results_if_needed(self, query: str, user_parsed: Dict[str, Any], 
                                  categories: List[str], initial_results: Dict[str, List[Document]],
                                  quality_assessment: Dict[str, Dict[str, Any]], total_needed: int) -> Dict[str, List[Document]]:
        """
        Augment results with external API calls if needed
        """
        final_results = initial_results.copy()
        
        for category in categories:
            if category == "날씨":
                # Handle weather separately
                final_results[category] = self._get_weather_info(user_parsed.get("region"))
                continue
                
            assessment = quality_assessment.get(category, {})
            existing_docs = final_results.get(category, [])
            # 중복 체크용 title/contentid set
            existing_titles = set(doc.metadata.get("title") for doc in existing_docs if doc.metadata.get("title"))
            existing_ids = set(doc.metadata.get("contentid") for doc in existing_docs if doc.metadata.get("contentid"))
            shortfall = total_needed - len(existing_docs)
            if assessment.get("needs_augmentation", False) and shortfall > 0:
                logger.info(f"Augmenting results for category: {category}")
                
                # Fetch external data
                external_data = self._fetch_external_data(category, user_parsed)
                
                if external_data:
                    # 중복 제거 (title 또는 contentid 기준)
                    unique_external = [
                        item for item in external_data
                        if (
                            (item.get("contentid") and item.get("contentid") not in existing_ids) or
                            (item.get("title") and item.get("title") not in existing_titles)
                        )
                    ][:shortfall]
                    external_docs = self._convert_to_documents(unique_external, category)
                    
                    # Add to VectorDB for future use if enabled
                    if self.enable_db_updates and self.db_updater and unique_external:
                        try:
                            db_docs = self.db_updater.create_documents_from_api_data(unique_external, category, query)
                            self.db_updater.add_documents_to_db(db_docs, category)
                        except Exception as e:
                            logger.error(f"Error updating VectorDB: {str(e)}")
                    
                    # Merge with existing results
                    final_results[category] = existing_docs + external_docs
                    
                    logger.info(f"Added {len(external_docs)} external results for {category}")
        
        return final_results
    
    def _fetch_external_data(self, category: str, user_parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch data from external APIs based on category"""
        if category in self.external_api_mapping:
            try:
                return self.external_api_mapping[category](user_parsed)
            except Exception as e:
                logger.error(f"Error fetching external data for {category}: {str(e)}")
                return []
        return []
    
    def _fetch_tourist_attractions(self, user_parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch tourist attractions from external API"""
        try:
            if user_parsed.get("region"):
                results = fetch_pet_friendly_places_only(user_parsed, limit=self.max_external_results)
                logger.info(f"Fetched {len(results)} tourist attractions from external API")
                return results
        except Exception as e:
            logger.error(f"Error fetching tourist attractions: {str(e)}")
        return []
    
    def _fetch_accommodations(self, user_parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch accommodation data from external API"""
        # This can be implemented with another API
        # For now, using the same API as tourist attractions
        try:
            if user_parsed.get("region"):
                results = fetch_pet_friendly_places_only(user_parsed, limit=self.max_external_results)
                # Filter for accommodations if possible
                accommodations = [r for r in results if any(keyword in r.get('title', '').lower() 
                                                        for keyword in ['호텔', '펜션', '리조트', '게스트하우스', '숙박'])]
                logger.info(f"Fetched {len(accommodations)} accommodations from external API")
                return accommodations
        except Exception as e:
            logger.error(f"Error fetching accommodations: {str(e)}")
        return []
    
    def _get_weather_info(self, region: str) -> List[Document]:
        """Get weather information"""
        if not region:
            return [Document(page_content="지역 정보가 없어 날씨를 조회할 수 없습니다.", metadata={})]
        
        try:
            weather_data = get_weather(region)
            current_time = get_current_time()
            
            if "error" not in weather_data:
                content = f"""
                        ### 현재 시간 및 날씨 정보
                        - 현재 시간: {current_time['full_datetime']}
                        - 도시: {weather_data['city']}
                        - 기온: {weather_data['temperature']}°C
                        - 습도: {weather_data['humidity']}%
                        - 강수형태: {weather_data['precipitation_type']}
                        - 풍속: {weather_data['wind_speed']} m/s
                """.strip()
                return [Document(page_content=content, metadata=weather_data)]
            else:
                return [Document(page_content=f"날씨 정보 조회 실패: {weather_data['error']}", metadata={})]
        except Exception as e:
            logger.error(f"Error getting weather info: {str(e)}")
            return [Document(page_content="날씨 정보를 가져오는데 실패했습니다.", metadata={})]
    
    def _convert_to_documents(self, external_data: List[Dict[str, Any]], category: str) -> List[Document]:
        """Convert external API data to Document objects"""
        documents = []
        
        for item in external_data:
            # Create content string
            content_parts = []
            if item.get('title'):
                content_parts.append(f"**{item['title']}**")
            if item.get('addr1'):
                content_parts.append(f"주소: {item['addr1']}")
            if item.get('tel'):
                content_parts.append(f"연락처: {item['tel']}")
            if item.get('pet_info'):
                content_parts.append(f"반려동물 정보: {item['pet_info']}")
            
            content = "\n".join(content_parts)
            
            # Add source information
            item['data_source'] = 'external_api'
            item['fetch_time'] = datetime.now().isoformat()
            item['category'] = category
            
            documents.append(Document(page_content=content, metadata=item))
        
        return documents
    
    def _generate_response(self, query: str, user_parsed: Dict[str, Any], 
                          results: Dict[str, List[Document]], stream: bool = False) -> str:
        """Generate final response using LLM"""
        
        # Prepare content sections
        content_sections = []
        
        for category, docs in results.items():
            if not docs:
                continue
                
            category_content = f"### {category} 정보\n"
            
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                
                # Generate map link
                place_name = metadata.get("title", f"장소 {i}")
                map_link = get_naver_map_link(place_name) if place_name != f"장소 {i}" else "#"
                
                place_info = f"**{i}. [{place_name}]({map_link})**\n"
                place_info += doc.page_content
                
                if metadata.get('data_source') == 'external_api':
                    place_info += "\n   *(최신 정보)*"
                
                place_info += "\n\n"
                category_content += place_info
            
            content_sections.append(category_content)
        
        content = "\n".join(content_sections)
        
        # Generate final response using LLM
        template = """
        당신은 반려동물과 함께하는 여행 전문 도우미입니다. 
        사용자의 질문에 대해 제공된 실제 데이터를 바탕으로 친절하고 유용한 답변을 제공해주세요.
        
        **중요 지침:**
        1. 제공된 데이터만을 사용하여 답변하세요
        2. 마크다운 형식으로 정리해서 응답하세요
        3. 각 장소의 네이버 지도 링크를 포함하세요
        4. 반려동물 관련 정보가 있다면 강조해서 안내하세요
        5. 여행 일정이나 코스 추천이 가능하다면 제안해주세요
        
        사용자 질문: {query}
        지역: {region}
        반려동물: {pet_type}
        여행 기간: {days}
        
        제공된 정보:
        {content}
        
        위 정보를 바탕으로 친절하고 상세한 답변을 제공해주세요:
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
        
        if stream:
            return chain.stream(inputs)
        else:
            return chain.invoke(inputs)


# Convenience functions for backward compatibility
def process_query(query: str, stream: bool = False) -> str:
    """
    Process query using the enhanced retriever system
    
    Args:
        query: User input query
        stream: Whether to stream the response
        
    Returns:
        Generated response
    """
    retriever = Retriever()
    return retriever.process_query(query, stream)


# # Example usage and testing
if __name__ == "__main__":
    # 테스트용 쿼리 (여행 일수별)
    test_queries = [
        "서울에서 강아지랑 1박 2일 여행 코스 추천해줘",  # days=2
        "부산에 고양이랑 3일 여행 가고 싶어",           # days=3
        "제주도 반려견과 1일 여행지 알려줘",             # days=1
        "강릉에서 2박 3일 강아지랑 여행 코스 짜줘"      # days=3
    ]
    retriever = Retriever()
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        try:
            result = retriever.process_query(query, stream=False)
            print(result)
        except Exception as e:
            print(f"Error: {str(e)}") 