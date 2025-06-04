import streamlit as st
import base64
import re
import random
import sys
import os
from agent.chatbot_agent import process_query, check_greeting

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 페이지 설정
st.set_page_config(page_title="🐶우리개 어디가?🐶", layout="wide")

# 스타일
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
        color: #333333;
    }
    .chat-container {
        width: 100%;
        padding: 0;
        margin: 0;
    }
    .bubble {
        display: inline-block;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-radius: 15px;
        font-size: 15px;
        max-width: 60%;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: 'Nanum Gothic', sans-serif;
        line-height: 1.4;
        position: relative;
        z-index: 1;
    }
    .bubble.left {
        background-color: #e1f7e1;
        color: #000000;
        float: left;
        margin-left: 60px;
    }
    .bubble.right {
        background-color: #d7ebfa;
        color: #000000;
        float: right;
        margin-right: 60px;
    }
    .chat-line {
        display: block;
        margin-bottom: 25px;
        margin-top: 45px;
        position: relative;
        width: 100%;
        overflow: visible;
        clear: both;
        min-height: 60px;
    }
    .profile-container {
        position: absolute;
        width: 50px;
        height: 50px;
        top: 0;
    }
    .profile-container.left {
        left: 0;
    }
    .profile-container.right {
        right: 0;
    }
    .profile-img {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        position: absolute;
        z-index: 3;
    }
    .character {
        width: 60px;
        height: 60px;
        position: absolute;
        z-index: 2;
    }
    .character.left {
        left: 35px;
        top: -45px;
    }
    .character.right {
        right: 35px;
        top: -45px;
    }
    .typing-indicator {
        background-color: #e1f7e1;
        padding: 8px 12px;
        border-radius: 15px;
        margin-left: 60px;
        display: inline-block;
        color: #666;
        font-size: 14px;
        animation: pulse 1.5s infinite;
        position: relative;
        z-index: 1;
    }
    .markdown-body {
        background-color: #e1f7e1;
        border-radius: 15px;
        padding: 12px 16px;
        margin-left: 60px;
        max-width: 60%;
        font-size: 15px;
        line-height: 1.4;
        position: relative;
        z-index: 1;
        float: left;
    }
    /* Streamlit 기본 컨테이너 스타일 재정의 */
    .main .block-container {
        max-width: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    .stMarkdown {
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    /* 사이드바 스타일 */
    .css-1d391kg {
        width: 14rem;
    }
    </style>
""", unsafe_allow_html=True)

# base64 이미지 로딩 함수
def img_to_b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

# 이미지 인코딩
user_profile_b64 = img_to_b64("./assets/user.png")
bot_profile_b64 = img_to_b64("./assets/bot.png")
user_dog_b64 = img_to_b64("./assets/foruser.png")
bot_dog_b64 = img_to_b64("./assets/forbot.png")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_bot" not in st.session_state:
    st.session_state.pending_bot = None
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# 사이드바 버튼 클릭 핸들러
def set_user_input(text):
    st.session_state.user_input = text

# 사이드바
with st.sidebar:
    logo_b64 = img_to_b64("./assets/logo.png")
    if logo_b64:
        st.sidebar.markdown(f'<img src="data:image/png;base64,{logo_b64}" width="200">', unsafe_allow_html=True)
    st.sidebar.title("반려견 여행 도우미")
    st.sidebar.write("반려견 여행 전문가와의 채팅입니다. 궁금한 점을 입력해보세요!")
    
    # 구분선
    st.sidebar.markdown("---")
    
    # 이용 가이드 섹션
    st.sidebar.markdown("### 💡 이용 가이드")
    
    # 주요 기능 목록
    st.sidebar.markdown("#### 🎯 주요 기능:")
    st.sidebar.markdown("""
    - 🗺️ 여행지 추천
    - 🏨 숙박 시설 안내
    - 🚌 대중교통 규정
    - ☀️ 날씨 정보
    - 🐕 반려동물 동반 정보
    """)
    
    # 구분선
    st.sidebar.markdown("---")
    
    # 인기 질문 섹션
    st.sidebar.markdown("### 🔥 인기 질문")
    
    # 인기 질문 버튼들
    if st.sidebar.button("속초 여행 추천해줘", key="sokcho", help="속초 여행 정보를 얻어보세요"):
        st.session_state.user_input = "속초 여행 추천해줘"
        st.rerun()
    
    if st.sidebar.button("제주도 반려동물 호텔", key="jeju", help="제주도의 반려동물 동반 호텔 정보를 얻어보세요"):
        st.session_state.user_input = "제주도 반려동물 동반 가능한 호텔 추천해줘"
        st.rerun()
    
    if st.sidebar.button("KTX 기차 반려동물 탑승", key="train", help="KTX 기차 이용시 반려동물 동반 규정을 확인해보세요"):
        st.session_state.user_input = "KTX 기차 반려동물 탑승 관련해서 알려줘"
        st.rerun()

# 타이틀
st.title("🐶우리개 어디가?🐶")

# 채팅 컨테이너
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        is_markdown = msg.get("is_markdown", False)
        
        try:
            content = content.encode("utf-16", "surrogatepass").decode("utf-16")
        except:
            content = re.sub(r"[\ud800-\udfff]", "", content)
        
        if role == "user":
            profile_b64 = user_profile_b64
            overlay_b64 = user_dog_b64
            position = "right"
        else:
            profile_b64 = bot_profile_b64
            overlay_b64 = bot_dog_b64
            position = "left"

        if not is_markdown:
            st.markdown(f"""
                <div class="chat-line">
                    <div class="profile-container {position}">
                        <img src="data:image/png;base64,{profile_b64}" class="profile-img">
                        <img src="data:image/png;base64,{overlay_b64}" class="character {position}">
                    </div>
                    <div class="bubble {position}">{content}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-line">
                    <div class="profile-container {position}">
                        <img src="data:image/png;base64,{profile_b64}" class="profile-img">
                        <img src="data:image/png;base64,{overlay_b64}" class="character {position}">
                    </div>
                    <div class="{'markdown-body' if role == 'bot' else f'bubble {position}'}">{content}</div>
                </div>
            """, unsafe_allow_html=True)

    # 타이핑 표시기
    if st.session_state.pending_bot:
        st.markdown(f"""
            <div class="chat-line">
                <div class="profile-container left">
                    <img src="data:image/png;base64,{bot_profile_b64}" class="profile-img">
                </div>
                <div class="typing-indicator"> 답변을 작성중입니다...</div>
            </div>
        """, unsafe_allow_html=True)

# 입력창
if "user_input" in st.session_state and st.session_state.user_input:
    user_input = st.session_state.user_input
    st.session_state.user_input = ""
else:
    user_input = st.chat_input("질문을 입력하세요...", key="chat_input")

# 입력 처리
if user_input:
    # 일반적인 대화 체크
    greeting_response = check_greeting(user_input)
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    if greeting_response:
        # 일반적인 대화인 경우 바로 응답
        st.session_state.messages.append({"role": "bot", "content": greeting_response})
        st.rerun()
    else:
        # 여행 관련 질문인 경우 process_query 실행
        st.session_state.pending_bot = user_input
        st.rerun()

if st.session_state.pending_bot:
    try:
        with st.spinner(''):
            bot_answer = process_query(st.session_state.pending_bot, stream=False)
            # 연속된 줄바꿈 제어
            bot_answer = re.sub(r'\n{3,}', '\n\n', bot_answer)  # 3개 이상 연속된 줄바꿈을 2개로 제한
            bot_answer = bot_answer.strip()  # 앞뒤 공백 제거
            
            # 마크다운 형식 메시지 추가
            st.session_state.messages.append({
                "role": "bot",
                "content": bot_answer,
                "is_markdown": True  # 마크다운 형식임을 표시
            })
    except Exception as e:
        error_message = f"죄송합니다. 응답 생성 중 오류가 발생했습니다. 다시 시도해주세요.\n\n오류: {str(e)}"
        st.session_state.messages.append({"role": "bot", "content": error_message})
        st.error(f"처리 중 오류 발생: {str(e)}")
    finally:
        st.session_state.pending_bot = None
        st.rerun()

transport_keywords = {
    "기차": ["기차", "ktx", "열차", "철도", "korail", "코레일"],
    "버스": ["버스", "시내버스", "시외버스", "고속버스"],
    "지하철": ["지하철", "전철", "metro"],
    "택시": ["택시", "call", "콜택시"]
}

# ### 🚄 기차 이용 안내

# #### KTX/일반열차 이용 규정
# [구체적인 규정]

# #### 이용 시 주의사항
# [주의사항 목록]

# #### 준비물 안내
# [필요한 준비물 목록]