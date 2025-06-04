from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import json
import vector_manger as vm

DEVICE = vm.is_mps_device()

# JSON 데이터 불러오기 
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 문서 생성 
def build_documents(data):
    documents =[] 
    for entry in data:
        content = entry["content"]
        metadata = entry.get("metadata", {})
        documents.append(Document(page_content=content, metadata=metadata))
    return documents 


# kure_v1 임베딩 FAISS 저장 
def build_faiss_index(documents, save_path):
    model = HuggingFaceEmbeddings(
        model_name="nlpai-lab/KURE-v1",
        model_kwargs = {'device':DEVICE}

    )
    db = FAISS.from_documents(documents, model)
    db.save_local(save_path)
    return db


if __name__ == "__main__":
    # 숙소 ---------------------------------------------------
    json_path = "../data/json/pet_lodging_places_202412.json"
    save_path = "../data/db/faiss/faiss_pet_kure"
    # 장소 ---------------------------------------------------
    # json_path = '../data/json/pet_friendly_places_2023.json'
    # save_path = '../data/db/faiss/faiss_place_kure'
    # 규정 ---------------------------------------------------
    # json_path  = '../data/json/pet_travel_vector_records_with_id.json'
    # save_path = '../data/db/faiss/faiss_regular_kure'
    data = load_json(json_path)
    documents = build_documents(data)
    db = build_faiss_index(documents, save_path)

    print(f"✅ kure_v1로 {len(documents)}개 문서를 임베딩하고 저장했습니다: {save_path}")