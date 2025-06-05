from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import vector_manger as vm 
import os 
from datetime import date
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from weather import get_weather
from typing import List, Dict, Any, Optional
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, CommaSeparatedListOutputParser
import logging
import requests
from naver_map_utils import NaverMapUtils

#-------------- LOAD -----------------

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# 지연 로딩을 위한 함수
def get_db(name: str):
    try:
        return vm.load_db(name)
    except Exception as e:
        print(f"Warning: Failed to load database {name}: {str(e)}")
        return None

# 데이터베이스 초기화를 함수 호출 시점으로 지연
db_pet = None
db_place = None

def ensure_db_loaded():
    global db_pet, db_place
    if db_pet is None:
        db_pet = get_db("faiss_pet_kure")
    if db_place is None:
        db_place = get_db("faiss_place_kure")

#------------------------------------- 

def get_naver_map_link(place_name: str) -> str:
    """
    네이버 지도 링크 생성 래퍼 함수
    
    Args:
        place_name (str): 장소 이름
        
    Returns:
        str: 네이버 지도 링크
    """
    return NaverMapUtils.get_map_link(place_name)

def get_category(query: str) -> List[str]:
    """
    질문을 받으면 카테고리에 맞는 리스트 추출 
    
    Args:
        query (str): 질의문 

    Returns:
        List[str]: List [카테고리]
    """
    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()
    
    category_prompt = PromptTemplate.from_template(
    """
    질문을 보고 해당되는 카테고리를 모두 골라서 콤마(,)로 구분해서 작성해줘 (복수 선택 가능):
    - 관광지
    - 숙박
    - 대중교통
    - 날씨
    
    {format_instructions}
    
    질문: {input}
    
    응답 예시: 
    input: "강릉으로 여행 가려고하는데 날씨가 괜찮을까?"
    output: 날씨
    
    input: "부산에서 기차나 버스에 반려견 태울 수 있어?"
    output: 대중교통
    
    input: "이번 주말에 버스타고 속초 가서 하루 자고 오고 싶어. 강아지랑 같이 갈 수 있을까?"
    output: 관광지, 숙박, 대중교통
    """
    )
    
    client = ChatOpenAI(
        model='gpt-4o-mini',
        api_key=openai_api_key,
        temperature=0
    )
    chain = category_prompt | client | output_parser
    result = chain.invoke({"input": query, "format_instructions": format_instructions})
    return result


def get_user_parser(query : str) -> Dict[str, Any]:
    
    user_parser_prompt = PromptTemplate.from_template("""
    당신은 사용자의 여행 요청 문장에서 다음 3가지를 정확히 추출해야 합니다.
    1. 여행 지역 이름 (예: 강릉, 제주도)
    2. 반려동물 종류 (예: 강아지, 고양이 등)
    3. 여행 일수
    - "이번 달 말","이번 달","다음 주"는 날짜 기준이 아닌 경우는 숫자로 바꾸지 말고 그대로 문자열로 출력하세요.
    - "주말"은 날짜 기준이 아닌 경우는 숫자로 바꾸지 말고 그대로 문자열로 출력하세요.
    - "글피"처럼 날짜 기준이 아닌 경우는 숫자로 바꾸지 말고 그대로 문자열로 출력하세요.
    - "당일치기"는 날짜 기준이 아닌 경우는 숫자로 바꾸지 말고 그대로 문자열로 출력하세요.
    - 날짜 기준이 아닌 경우는 숫자로 바꾸지 말고 그대로 문자열로 출력하세요.
    - 명확하지 않으면 "null"을 사용하세요.

    출력 형식(JSON):
    {format_instructions}

    예시 입력 1:
    제주도에 고양이랑 2박 3일 놀러 가려고 해
    예시 출력 1:
    {{"region": "제주도", "pet_type": "고양이", "days": 3}}

    예시 입력 2:
    강아지랑 강릉으로 다음 주에 여행 가고 싶어
    예시 출력 2:
    {{"region": "강릉", "pet_type": "강아지", "days": "다음 주"}}

    예시 입력 3:
    양양에 반려견 두 마리랑 모레부터 4일간 머물 곳 있을까?
    예시 출력 3:
    {{"region": "양양", "pet_type": "반려견", "days": 4}}

    예시 입력 4:
    이번 주말에 고양이랑 단양 여행 가면 어때?
    예시 출력 4:
    {{"region": "단양", "pet_type": "고양이", "days": "주말"}}

    예시 입력 5:
    글피에 강아지랑 속초로 여행 갈 건데
    예시 출력 5:
    {{"region": "속초", "pet_type": "강아지", "days": "글피"}}
        
    예시 입력 6:
    이번 주에 수요일에 고양이랑 단양 여행 가면 어때?
    예시 출력 6:
    {{"region": "단양", "pet_type": "고양이", "days": "당일치기"}}

    이제 아래 사용자 입력을 분석해 주세요:

    입력:
    {query}
    """)

    
    #  스키마 정의 
    response_schemas = [
        ResponseSchema(name="region", description="여행 지역 이름"),
        ResponseSchema(name= "pet_type", description ="반려동물 종류"),
        ResponseSchema(name="days", description="여행 일수 (숫자만)")
    ]
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = parser.get_format_instructions()
    
    # chat_gpt 정의 
    llm = ChatOpenAI(
        model ='gpt-4o-mini', 
        api_key=openai_api_key,
        temperature= 0
    )
    
    chain = user_parser_prompt | llm | parser
    output = chain.invoke({
        "query" : query,
        "format_instructions" : format_instructions
    })
    return output

# 네이버 지도 장소 유효성 확인 
def is_valid_place(place_name: str) -> bool:
    url = 'https://openapi.naver.com/v1/search/local.json'
    params = {
        'query' : place_name,
    }
    headers = {
        'X-Naver-Client-Id' : os.getenv('NAVER_CLIENT_KEY'),
        'X-Naver-Client-Secret' : os.getenv('NAVER_CLIENT_SECRET_KEY')
    }
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    return len(data['items']) > 0






# if __name__ == '__main__':
#     print(is_valid_place("속초 냥냥냥카페"))
    # query = "속초에 강아지랑 2박 3일 가고 싶어"
    # parsed = get_user_parser(query)
    # print(parsed)
