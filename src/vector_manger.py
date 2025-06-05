import pathlib, functools, torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from typing import Dict, List, Sequence, Optional, Tuple
from langchain.schema import Document
import logging 
import ast
import os

# Initialize device at module level
_DEVICE = None
def _initialize_device():
    global _DEVICE
    if _DEVICE is None:
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                _DEVICE = "mps"
            elif torch.cuda.is_available():
                _DEVICE = "cuda"
            else:
                _DEVICE = "cpu"
            logging.info(f"Using device: {_DEVICE}")
        except Exception as e:
            logging.warning(f"Error initializing device, falling back to CPU: {str(e)}")
            _DEVICE = "cpu"
    return _DEVICE

# 벡터 스코어 로그 
_db_cache: Dict[str, FAISS] = {}
category_to_db: Dict[str, str] = {
    "관광지": "faiss_place_kure",
    "숙박":   "faiss_pet_kure",
    "대중교통": "faiss_regular_kure",
}

def get_device():
    """Get the appropriate device for computation"""
    device = _initialize_device()
    logging.info(f"Current device: {device}")
    return device

# 싱글 임베딩
@functools.lru_cache(maxsize=1)
def get_embedding(device: Optional[str] = None):
    if device is None:
        device = get_device()
    
    device_map = {
        "cuda": 0,
        "mps": "mps",
        "cpu": -1
    }
    
    device_id = device_map.get(device, -1)
    
    return HuggingFaceEmbeddings(
        model_name="nlpai-lab/KURE-v1",
        model_kwargs={"device": device_id},
        encode_kwargs={"normalize_embeddings": True}
    )

def get_project_root():
    """프로젝트 루트 디렉토리 경로를 반환합니다."""
    current_file = pathlib.Path(__file__).resolve()
    return current_file.parent.parent

def load_db(name: str) -> FAISS:
    """
    FAISS 데이터베이스를 로드합니다.
    데이터베이스가 이미 캐시되어 있다면 캐시된 버전을 반환합니다.
    """
    if name in _db_cache:
        logging.info(f"Using cached database: {name}")
        return _db_cache[name]
    
    try:
        root_dir = get_project_root()
        db_path = root_dir / "data" / "db" / "faiss" / name
        
        if not db_path.exists():
            logging.error(f"Database directory not found: {db_path}")
            raise FileNotFoundError(f"Database directory not found: {db_path}")
            
        logging.info(f"Loading database from: {db_path}")
        db = FAISS.load_local(
            folder_path=str(db_path),
            embeddings=get_embedding(),
            allow_dangerous_deserialization=True,
        )
        logging.info(f"Successfully loaded database: {name}")
        _db_cache[name] = db
        return db
    except Exception as e:
        logging.error(f"Error loading database {name}: {str(e)}")
        raise

def multiretrieve_by_category(
    query: str,
    categories: Sequence[str] | str,
    *,
    k_each: int = 5,
    top_k: int = 5,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, List[Document]]:
    """
    카테고리별로 문서를 검색합니다.
    날씨 카테고리는 DB 검색에서 제외되며, 호출측에서 별도 처리해야 합니다.
    """
    if not query or not isinstance(query, str):
        logging.error("Invalid query: query must be a non-empty string")
        raise ValueError("Query must be a non-empty string")

    # ── 1. 문자열이면 파싱해서 리스트로 변환 ─────────────────────
    if isinstance(categories, str):
        try:
            categories = ast.literal_eval(categories)   # '["관광지", "숙박"]' → ["관광지","숙박"]
        except Exception:
            # 콤마로만 구분된 단순 문자열 "관광지,숙박"
            categories = [c.strip() for c in categories.split(",") if c.strip()]

    # 이제부터는 리스트가 보장됨
    # 날씨 카테고리는 DB 검색에서 제외 향후 개선 코드로 수정 예정 
    db_categories = [c for c in categories if c != "날씨"]
    
    if not db_categories:
        logging.warning("No valid categories for DB search")
        return {}

    results: Dict[str, List[Document]] = {}
    for cat in db_categories:
        try:
            if cat not in category_to_db:
                logging.warning(f"Unsupported category: {cat}")
                continue
                
            logging.info(f"Searching for category: {cat}")
            db = load_db(category_to_db[cat])
            docs_scores: List[Tuple[Document, float]] = db.similarity_search_with_score(query, k=k_each)

            w = 1.0 if weights is None else weights.get(cat, 1.0)
            ranked = sorted(
                (((1 - score) * w, doc) for doc, score in docs_scores),
                key=lambda x: x[0],
                reverse=True,
            )[:top_k]

            results[cat] = [doc for _, doc in ranked]
            logging.info(f"Found {len(results[cat])} results for category: {cat}")
            
        except Exception as e:
            logging.error(f"Error processing category {cat}: {str(e)}")
            results[cat] = []

    return results


def list_loaded() -> List[str]:
    """현재 메모리에 로드 된 DB 이름"""
    return list(_db_cache.keys())

def is_mps_device():
    """Check if MPS (Metal Performance Shaders) is available"""
    try:
        return 'mps' if (torch.backends.mps.is_available() and torch.backends.mps.is_built()) else 'cpu'
    except Exception as e:
        logging.warning(f"Error checking MPS availability, falling back to CPU: {str(e)}")
        return 'cpu'