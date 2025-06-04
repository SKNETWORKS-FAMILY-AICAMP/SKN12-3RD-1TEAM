from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import vector_manger as vm 
device  = vm.is_mps_device()

# 임베딩 모델 로드 
embedding_model = HuggingFaceEmbeddings(
    model_name = 'nlpai-lab/KURE-v1',
    model_kwargs = {'device': device}
)

# FAISS DB로드

db = FAISS.load_local(
    folder_path='../data/db/faiss/faiss_pet_kure',
    embeddings=embedding_model,
    allow_dangerous_deserialization= True # 신뢰성 명시
)


# 쿼리 실행 

query = '부산에서 반려견과 함꼐 묵을 수 있는 숙소'

result = db.similarity_search(query, k=5)

# 결과 출력
print(f'질문 : {query}')

for i, doc in enumerate(result,1):
    print(f"\n[{i}] ✅ 유사 문서:\n{doc.page_content}")
    print(f"📎 메타데이터: {doc.metadata}")