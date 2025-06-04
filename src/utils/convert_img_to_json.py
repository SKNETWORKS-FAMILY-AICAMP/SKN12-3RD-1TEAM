import re
import cv2
import numpy as np
import pytesseract
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os, glob, json
from typing import List, Dict

IMG_DIR = '../data/regular'
LANG_PACK = "kor+eng"
IMG_PATTERN = r"part(?P<part>\d{2})_(?P<idx>\d{2})"
OUTFILE = "../data/pet_travel_vector_records.json"


def preprocess_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 2)
    # Sharpening
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharp = cv2.filter2D(thresh, -1, kernel)
    return sharp

def clean_text(text: str) -> str:
    # 연속 개행/공백 정리
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    # 특수문자 정리 (한글, 영문, 숫자, 기본 문장부호만 남김)
    text = re.sub(r'[^\w\s.,!?~()\[\]{}:;\'\"@#&%\-–—+=/|\\]', '', text)
    text = text.strip()
    return text

def ocr_image(path: str) -> str:
    img = preprocess_image(path)
    # Tesseract config: --psm 6 (Assume a single uniform block of text)
    config = '--psm 6'
    text = pytesseract.image_to_string(img, lang=LANG_PACK, config=config)
    return clean_text(text)

def build_records() -> List[Dict]:
    records = []
    for f in sorted(glob.glob(os.path.join(IMG_DIR, '*'))):
        m = re.search(IMG_PATTERN, os.path.basename(f))
        if not m:
            continue
        part_id = f"part{m.group('part')}"
        slice_idx = int(m.group('idx'))
        record = {
            "content": ocr_image(f),
            "metadata": {
                "source": "대한민국 구석구석",
                "image_file": os.path.basename(f),
                "part_id": part_id,
                "slice_index": slice_idx,
            }
        }
        records.append(record)
    return records

if __name__ == "__main__":
    data = build_records()
    with open(OUTFILE, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(data)} records → {OUTFILE}")