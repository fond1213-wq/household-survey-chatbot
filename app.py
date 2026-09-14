import io
import os
import json
import hashlib
import importlib.metadata
from datetime import datetime

import streamlit as st
from pypdf import PdfReader


# ============================================================
# 기본 설정
# ============================================================

st.set_page_config(
    page_title="가구부문통계조사 챗봇",
    page_icon="📚",
    layout="wide"
)

st.title("📚 가구부문통계조사 챗봇")

st.caption(
    "가구부문 통계조사 업무자료를 기반으로 "
    "질문에 답변하는 RAG 챗봇"
)

st.divider()


# ============================================================
# 폴더
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

os.makedirs(
    DATA_DIR,
    exist_ok=True
)

INDEX_FILE = os.path.join(
    DATA_DIR,
    "documents.json"
)

CHUNKS_FILE = os.path.join(
    DATA_DIR,
    "chunks.json"
)

EMBEDDINGS_FILE = os.path.join(
    DATA_DIR,
    "embeddings.json"
)


# ============================================================
# Pillow
# ============================================================

try:

    from PIL import Image

    PIL_SUPPORT = True
    PIL_ERROR = ""

except Exception as e:

    PIL_SUPPORT = False
    PIL_ERROR = repr(e)


# ============================================================
# HEIC
# ============================================================

try:

    from pillow_heif import register_heif_opener

    register_heif_opener()

    HEIC_SUPPORT = True
    HEIC_ERROR = ""

except Exception as e:

    HEIC_SUPPORT = False
    HEIC_ERROR = repr(e)


# ============================================================
# NumPy
# ============================================================

try:

    import numpy as np

    NUMPY_SUPPORT = True
    NUMPY_ERROR = ""

except Exception as e:

    NUMPY_SUPPORT = False
    NUMPY_ERROR = repr(e)


# ============================================================
# Requests
# ============================================================

try:

    import requests

    REQUESTS_SUPPORT = True
    REQUESTS_ERROR = ""

except Exception as e:

    REQUESTS_SUPPORT = False
    REQUESTS_ERROR = repr(e)


# ============================================================
# RapidOCR
# ============================================================

try:

    from rapidocr import (
        RapidOCR,
        EngineType,
        LangDet,
        LangRec,
        ModelType,
        OCRVersion
    )

    OCR_SUPPORT = True
    OCR_IMPORT_ERROR = ""

except Exception as e:

    OCR_SUPPORT = False
    OCR_IMPORT_ERROR = repr(e)


# ============================================================
# Sentence Transformers
# ============================================================

try:

    from sentence_transformers import SentenceTransformer

    EMBEDDING_SUPPORT = True
    EMBEDDING_IMPORT_ERROR = ""

except Exception as e:

    EMBEDDING_SUPPORT = False
    EMBEDDING_IMPORT_ERROR = repr(e)


# ============================================================
# OpenAI
# ============================================================

try:

    from openai import OpenAI

    OPENAI_SUPPORT = True
    OPENAI_IMPORT_ERROR = ""

except Exception as e:

    OPENAI_SUPPORT = False
    OPENAI_IMPORT_ERROR = repr(e)


# ============================================================
# 버전
# ============================================================

try:
    RAPIDOCR_VERSION = importlib.metadata.version(
        "rapidocr"
    )
except Exception:
    RAPIDOCR_VERSION = "확인 불가"


try:
    ONNXRUNTIME_VERSION = importlib.metadata.version(
        "onnxruntime"
    )
except Exception:
    ONNXRUNTIME_VERSION = "확인 불가"


try:
    SENTENCE_TRANSFORMERS_VERSION = (
        importlib.metadata.version(
            "sentence-transformers"
        )
    )
except Exception:
    SENTENCE_TRANSFORMERS_VERSION = "확인 불가"


# ============================================================
# 임베딩 모델
# ============================================================

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-mpnet-base-v2"
)


# ============================================================
# OCR 모델
# ============================================================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


DET_MODEL = os.path.join(
    MODEL_DIR,
    "ch_PP-OCRv5_det_mobile.onnx"
)

REC_MODEL = os.path.join(
    MODEL_DIR,
    "korean_PP-OCRv5_rec_mobile.onnx"
)

DICT_FILE = os.path.join(
    MODEL_DIR,
    "ppocrv5_korean_dict.txt"
)


# ============================================================
# OCR 모델 URL
# ============================================================

MODEL_URLS = {

    "det":
        "https://www.modelscope.cn/models/"
        "RapidAI/RapidOCR/resolve/v3.9.2/"
        "onnx/PP-OCRv5/det/"
        "ch_PP-OCRv5_det_mobile.onnx",

    "rec":
        "https://www.modelscope.cn/models/"
        "RapidAI/RapidOCR/resolve/v3.9.2/"
        "onnx/PP-OCRv5/rec/"
        "korean_PP-OCRv5_rec_mobile.onnx",

    "dict":
        "https://www.modelscope.cn/models/"
        "RapidAI/RapidOCR/resolve/v3.9.2/"
        "paddle/PP-OCRv5/rec/"
        "korean_PP-OCRv5_rec_mobile/"
        "ppocrv5_korean_dict.txt"
}


# ============================================================
# 자료 목록
# ============================================================

def load_documents():

    if not os.path.exists(
        INDEX_FILE
    ):
        return []

    try:

        with open(
            INDEX_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if isinstance(
            data,
            list
        ):
            return data

    except Exception:

        pass

    return []


# ============================================================
# 자료 목록 저장
# ============================================================

def save_documents(
    documents
):

    with open(
        INDEX_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            documents,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# 임베딩 저장
# ============================================================

def load_embeddings():

    if not os.path.exists(
        EMBEDDINGS_FILE
    ):
        return {}

    try:

        with open(
            EMBEDDINGS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return {}


def save_embeddings(
    embeddings
):

    with open(
        EMBEDDINGS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            embeddings,
            f,
            ensure_ascii=False
        )


# ============================================================
# 파일 ID
# ============================================================

def make_file_id(
    filename,
    content
):

    h = hashlib.sha256()

    h.update(
        filename.encode(
            "utf-8",
            errors="ignore"
        )
    )

    h.update(
        content
    )

    return h.hexdigest()[:16]


# ============================================================
# 임베딩 모델
# ============================================================

@st.cache_resource
def load_embedding_model():

    if not EMBEDDING_SUPPORT:

        raise RuntimeError(
            "sentence-transformers import 실패:\n"
            + EMBEDDING_IMPORT_ERROR
        )

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    return model


# ============================================================
# 텍스트 임베딩
# ============================================================

def create_embeddings(
    texts
):

    model = load_embedding_model()

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return vectors.tolist()


# ============================================================
# 문서 Chunk 생성
# ============================================================

def split_text_into_chunks(
    text,
    chunk_size=700,
    overlap=100
):

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    paragraphs = [
        p.strip()
        for p in text.split("\n")
        if p.strip()
    ]

    chunks = []

    current = ""


    for paragraph in paragraphs:

        # 페이지 표시가 있으면
        # 해당 페이지 정보를 보존
        if len(current) + len(paragraph) <= chunk_size:

            if current:

                current += "\n"

            current += paragraph

        else:

            if current:

                chunks.append(
                    current.strip()
                )

            # overlap
            previous = current[
                -overlap:
            ] if current else ""

            current = (
                previous
                + "\n"
                + paragraph
            ).strip()


    if current:

        chunks.append(
            current.strip()
        )


    # 너무 짧은 chunk 제거
    chunks = [
        c for c in chunks
        if len(c.strip()) >= 20
    ]

    return chunks


# ============================================================
# 전체 Chunk 생성
# ============================================================

def rebuild_chunks():

    documents = load_documents()

    all_chunks = []


    for document in documents:

        text_filename = document.get(
            "text_file"
        )

        if not text_filename:
            continue


        text_path = os.path.join(
            DATA_DIR,
            text_filename
        )

        if not os.path.exists(
            text_path
        ):
            continue


        try:

            with open(
                text_path,
                "r",
                encoding="utf-8"
            ) as f:

                text = f.read()

        except Exception:

            continue


        chunks = split_text_into_chunks(
            text
        )


        for index, chunk in enumerate(
            chunks
        ):

            chunk_id = (
                document.get(
                    "id",
                    ""
                )
                + "_"
                + str(index)
            )


            all_chunks.append(
                {
                    "id": chunk_id,

                    "document_id":
                        document.get(
                            "id",
                            ""
                        ),

                    "filename":
                        document.get(
                            "filename",
                            ""
                        ),

                    "type":
                        document.get(
                            "type",
                            ""
                        ),

                    "chunk_index":
                        index,

                    "text":
                        chunk
                }
            )


    with open(
        CHUNKS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_chunks,
            f,
            ensure_ascii=False,
            indent=2
        )


    return all_chunks


# ============================================================
# Chunk 불러오기
# ============================================================

def load_chunks():

    if not os.path.exists(
        CHUNKS_FILE
    ):

        return []


    try:

        with open(
            CHUNKS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return []


# ============================================================
# 임베딩 인덱스 재구축
# ============================================================

def rebuild_embedding_index():

    chunks = load_chunks()

    if not chunks:

        chunks = rebuild_chunks()


    if not chunks:

        return 0


    texts = [
        chunk["text"]
        for chunk in chunks
    ]


    vectors = create_embeddings(
        texts
    )


    embeddings = {}

    for chunk, vector in zip(
        chunks,
        vectors
    ):

        embeddings[
            chunk["id"]
        ] = vector


    save_embeddings(
        embeddings
    )


    return len(
        vectors
    )


# ============================================================
# 자료 등록
# ============================================================

def register_document(
    filename,
    doc_type,
    text,
    original_size=0,
    page_count=None
):

    if not text:

        return (
            False,
            "저장할 내용이 없습니다."
        )


    text = text.strip()


    if not text:

        return (
            False,
            "저장할 내용이 없습니다."
        )


    documents = load_documents()


    text_bytes = text.encode(
        "utf-8"
    )


    file_id = make_file_id(
        filename,
        text_bytes
    )


    # 중복 검사
    for document in documents:

        if document.get(
            "id"
        ) == file_id:

            return (
                False,
                "이미 등록된 자료입니다."
            )


    base_name = os.path.splitext(
        filename
    )[0]


    safe_name = (
        base_name
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


    text_filename = (
        safe_name
        + "_"
        + file_id
        + ".txt"
    )


    text_path = os.path.join(
        DATA_DIR,
        text_filename
    )


    with open(
        text_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            text
        )


    document = {

        "id":
            file_id,

        "filename":
            filename,

        "type":
            doc_type,

        "text_file":
            text_filename,

        "size":
            original_size,

        "page_count":
            page_count,

        "text_length":
            len(text),

        "registered_at":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    }


    documents.append(
        document
    )


    save_documents(
        documents
    )


    # 새 자료가 들어왔으므로
    # chunk/embedding을 다시 생성
    try:

        rebuild_chunks()

        rebuild_embedding_index()

    except Exception as e:

        return (
            True,
            "자료는 등록되었지만 "
            "임베딩 생성 중 오류가 발생했습니다.\n"
            + repr(e)
        )


    return (
        True,
        "자료가 정상적으로 등록되었습니다."
    )


# ============================================================
# PDF 텍스트 추출
# ============================================================

def extract_pdf_text(
    file_bytes
):

    reader = PdfReader(
        io.BytesIO(
            file_bytes
        )
    )


    full_text = ""


    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            text = page.extract_text()

        except Exception:

            text = None


        if text:

            text = text.strip()


            if text:

                full_text += (
                    "\n\n"
                    "===== 페이지 "
                    + str(page_number)
                    + " =====\n\n"
                    + text
                )


    return (
        full_text.strip(),
        len(reader.pages)
    )


# ============================================================
# OCR 모델 다운로드
# ============================================================

def download_model(
    url,
    path
):

    if os.path.exists(
        path
    ):

        try:

            if os.path.getsize(
                path
            ) > 0:

                return (
                    True,
                    "이미 존재"
                )

        except Exception:

            pass


    if not REQUESTS_SUPPORT:

        return (
            False,
            "requests가 설치되어 있지 않습니다."
        )


    try:

        response = requests.get(
            url,
            stream=True,
            timeout=180
        )


        response.raise_for_status()


        with open(
            path,
            "wb"
        ) as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:

                    f.write(
                        chunk
                    )


        if (
            not os.path.exists(path)
            or os.path.getsize(path) == 0
        ):

            return (
                False,
                "다운로드 파일이 비어 있습니다."
            )


        return (
            True,
            "다운로드 완료"
        )


    except Exception as e:

        try:

            if os.path.exists(
                path
            ):

                os.remove(
                    path
                )

        except Exception:

            pass


        return (
            False,
            repr(e)
        )


# ============================================================
# OCR 모델 준비
# ============================================================

@st.cache_resource
def prepare_ocr_models():

    results = []

    all_ok = True


    for name, url, path in [

        (
            "det",
            MODEL_URLS["det"],
            DET_MODEL
        ),

        (
            "rec",
            MODEL_URLS["rec"],
            REC_MODEL
        ),

        (
            "dict",
            MODEL_URLS["dict"],
            DICT_FILE
        )
    ]:

        success, message = (
            download_model(
                url,
                path
            )
        )


        results.append(
            (
                name,
                success,
                message
            )
        )


        if not success:

            all_ok = False


    return (
        all_ok,
        results
    )


# ============================================================
# OCR 엔진
# ============================================================

@st.cache_resource
def create_ocr_engine():

    if not OCR_SUPPORT:

        raise RuntimeError(
            "RapidOCR import 실패:\n"
            + OCR_IMPORT_ERROR
        )


    if not NUMPY_SUPPORT:

        raise RuntimeError(
            "NumPy import 실패:\n"
            + NUMPY_ERROR
        )


    for path in [
        DET_MODEL,
        REC_MODEL,
        DICT_FILE
    ]:

        if not os.path.exists(
            path
        ):

            raise FileNotFoundError(
                "OCR 모델 파일이 없습니다:\n"
                + path
            )


    params = {

        "Det.engine_type":
            EngineType.ONNXRUNTIME,

        "Det.lang_type":
            LangDet.CH,

        "Det.model_type":
            ModelType.MOBILE,

        "Det.ocr_version":
            OCRVersion.PPOCRV5,

        "Det.model_path":
            DET_MODEL,

        "Rec.engine_type":
            EngineType.ONNXRUNTIME,

        "Rec.lang_type":
            LangRec.KOREAN,

        "Rec.model_type":
            ModelType.MOBILE,

        "Rec.ocr_version":
            OCRVersion.PPOCRV5,

        "Rec.model_path":
            REC_MODEL,

        "Rec.rec_keys_path":
            DICT_FILE
    }


    try:

        return RapidOCR(
            params=params
        )

    except Exception as e:

        raise RuntimeError(
            "OCR 엔진 초기화 실패:\n"
            + repr(e)
        )


# ============================================================
# OCR 실행
# ============================================================

def run_ocr(
    pil_image
):

    try:

        image = pil_image.convert(
            "RGB"
        )


        max_dimension = 2500

        width, height = image.size


        if max(
            width,
            height
        ) > max_dimension:

            ratio = (
                max_dimension
                / max(
                    width,
                    height
                )
            )


            image = image.resize(
                (
                    int(width * ratio),
                    int(height * ratio)
                )
            )


        image_array = np.array(
            image
        )


        engine = create_ocr_engine()


        result = engine(
            image_array
        )


        return (
            result,
            None
        )


    except Exception as e:

        return (
            None,
            repr(e)
        )


# ============================================================
# OCR 결과 텍스트
# ============================================================

def extract_ocr_text(
    result
):

    texts = []


    try:

        if hasattr(
            result,
            "txts"
        ):

            for value in (
                result.txts or []
            ):

                if value is not None:

                    text = str(
                        value
                    ).strip()


                    if text:

                        texts.append(
                            text
                        )


            return texts


    except Exception:

        pass


    return texts


# ============================================================
# 의미 기반 검색
# ============================================================

def semantic_search(
    question,
    top_k=6
):

    chunks = load_chunks()

    embeddings = load_embeddings()


    if not chunks:

        return []


    if not embeddings:

        rebuild_embedding_index()

        embeddings = load_embeddings()


    model = load_embedding_model()


    query_vector = model.encode(
        question,
        normalize_embeddings=True
    )


    results = []


    for chunk in chunks:

        chunk_id = chunk.get(
            "id"
        )


        vector = embeddings.get(
            chunk_id
        )


        if vector is None:

            continue


        vector_array = np.array(
            vector,
            dtype=float
        )


        score = float(
            np.dot(
                query_vector,
                vector_array
            )
        )


        results.append(
            {
                "score":
                    score,

                "filename":
                    chunk.get(
                        "filename",
                        ""
                    ),

                "type":
                    chunk.get(
                        "type",
                        ""
                    ),

                "text":
                    chunk.get(
                        "text",
                        ""
                    ),

                "chunk_index":
                    chunk.get(
                        "chunk_index",
                        0
                    ),

                "document_id":
                    chunk.get(
                        "document_id",
                        ""
                    )
            }
        )


    results.sort(
        reverse=True,
        key=lambda x: x["score"]
    )


    return results[
        :top_k
    ]


# ============================================================
# OpenAI API 키
# ============================================================

def get_openai_api_key():

    try:

        if "OPENAI_API_KEY" in st.secrets:

            return st.secrets[
                "OPENAI_API_KEY"
            ]

    except Exception:

        pass


    return os.environ.get(
        "OPENAI_API_KEY",
        ""
    )


# ============================================================
# LLM 답변 생성
# ============================================================

def generate_answer(
    question,
    search_results
):

    api_key = get_openai_api_key()


    if not api_key:

        return (
            None,
            "OPENAI_API_KEY가 설정되지 않았습니다."
        )


    if not OPENAI_SUPPORT:

        return (
            None,
            "openai 패키지를 불러오지 못했습니다.\n"
            + OPENAI_IMPORT_ERROR
        )


    try:

        client = OpenAI(
            api_key=api_key
        )


        context_parts = []


        for index, result in enumerate(
            search_results,
            start=1
        ):

            context_parts.append(
                (
                    f"[자료 {index}]\n"
                    f"파일명: {result['filename']}\n"
                    f"자료유형: {result['type']}\n"
                    f"관련도: {result['score']:.4f}\n"
                    f"내용:\n{result['text']}"
                )
            )


        context = "\n\n".join(
            context_parts
        )


        system_prompt = """
당신은 대한민국 가구부문 통계조사 업무를 지원하는
전문 조사원 보조 AI입니다.

반드시 제공된 자료를 근거로 답변하십시오.

규칙:

1. 제공된 자료에 없는 내용을 사실처럼 만들어내지 마십시오.

2. 질문의 의도를 먼저 파악한 뒤 답변하십시오.

3. 단순히 질문의 단어가 포함된 문장을 복사하지 말고,
   관련된 여러 자료를 종합해서 설명하십시오.

4. 통계조사 업무와 관련된 판단 기준, 예외사항,
   조사대상기간, 정의 등이 자료에 있다면 함께 설명하십시오.

5. 자료에 근거가 부족하면
   "제공된 자료만으로는 정확히 판단하기 어렵습니다."
   라고 명확히 말하십시오.

6. 가능하면 답변을
   - 결론
   - 판단 기준
   - 예외/주의사항
   순서로 구성하십시오.

7. 답변 마지막에는 반드시
   "근거 자료"를 표시하십시오.

8. 근거 자료는 제공된 자료의 파일명을 사용하십시오.

9. 자료의 내용과 일반적인 상식을 혼동하지 마십시오.

10. 사용자가 조사원 입장에서 실제로 어떻게 판단해야 하는지
    묻는 경우에는 실무적으로 이해하기 쉽게 설명하십시오.
"""


        user_prompt = f"""
다음은 사용자의 질문입니다.

[질문]
{question}

다음은 검색된 통계조사 자료입니다.

{context}

위 자료를 근거로 질문에 답변하십시오.
"""


        response = client.responses.create(

            model="gpt-5-mini",

            instructions=system_prompt,

            input=user_prompt
        )


        answer = response.output_text


        return (
            answer,
            None
        )


    except Exception as e:

        return (
            None,
            repr(e)
        )


# ============================================================
# 사이드바
# ============================================================

with st.sidebar:

    st.header(
        "📂 메뉴"
    )


    menu = st.radio(
        "이동",
        [
            "질문하기",
            "자료관리"
        ]
    )


    st.divider()


    st.subheader(
        "⚙️ 시스템 상태"
    )


    if OCR_SUPPORT:

        st.success(
            "OCR: ✅"
        )

    else:

        st.error(
            "OCR: ❌"
        )


    if EMBEDDING_SUPPORT:

        st.success(
            "의미검색: ✅"
        )

    else:

        st.error(
            "의미검색: ❌"
        )


    if get_openai_api_key():

        st.success(
            "LLM: ✅ API 연결"
        )

    else:

        st.warning(
            "LLM: ⚠️ API 키 없음"
        )


    st.caption(
        f"RapidOCR {RAPIDOCR_VERSION}"
    )

    st.caption(
        f"Embedding {SENTENCE_TRANSFORMERS_VERSION}"
    )


    documents = load_documents()


    st.divider()


    st.metric(
        "📚 등록 자료",
        len(documents)
    )


# ============================================================
# 질문하기
# ============================================================

if menu == "질문하기":

    st.subheader(
        "💬 질문하기"
    )


    documents = load_documents()


    if not documents:

        st.info(
            "먼저 자료관리에서 "
            "PDF/TXT/사진 자료를 등록해주세요."
        )


    else:

        question = st.text_area(
            "궁금한 내용을 입력하세요.",
            placeholder=(
                "예: 가끔 돈을 받고 일하는 주부도 "
                "취업자로 조사해야 하나요?"
            ),
            height=130,
            key="question_input"
        )


        if st.button(
            "🤖 자료 기반으로 답변하기",
            use_container_width=True
        ):

            if not question.strip():

                st.warning(
                    "질문을 입력해주세요."
                )

            else:

                # ------------------------------------------------
                # 1. 의미 검색
                # ------------------------------------------------

                with st.spinner(
                    "질문의 의미를 분석하고 "
                    "관련 자료를 검색하고 있습니다..."
                ):

                    try:

                        search_results = semantic_search(
                            question,
                            top_k=6
                        )

                    except Exception as e:

                        search_results = []

                        st.error(
                            "의미 기반 검색 중 오류가 발생했습니다."
                        )

                        st.code(
                            repr(e)
                        )


                st.session_state[
                    "search_results"
                ] = search_results


                st.session_state[
                    "last_question"
                ] = question


                # ------------------------------------------------
                # 2. 검색 결과
                # ------------------------------------------------

                if search_results:

                    st.subheader(
                        "🔎 관련 자료"
                    )


                    for index, result in enumerate(
                        search_results,
                        start=1
                    ):

                        score = result[
                            "score"
                        ]


                        # 너무 낮은 관련도는 표시하지 않음
                        if score < 0.25:

                            continue


                        with st.expander(
                            f"{index}. "
                            f"{result['filename']} "
                            f"  |  관련도 {score:.3f}"
                        ):

                            st.caption(
                                f"자료 유형: "
                                f"{result['type']}"
                            )


                            st.write(
                                result["text"]
                            )


                # ------------------------------------------------
                # 3. LLM 답변
                # ------------------------------------------------

                st.divider()


                st.subheader(
                    "🤖 AI 답변"
                )


                if not search_results:

                    st.warning(
                        "질문과 관련된 자료를 찾지 못했습니다."
                    )


                else:

                    good_results = [
                        r for r in search_results
                        if r["score"] >= 0.25
                    ]


                    if not good_results:

                        st.warning(
                            "관련도가 충분히 높은 "
                            "자료를 찾지 못했습니다."
                        )


                    elif not get_openai_api_key():

                        st.info(
                            "현재 의미 기반 검색까지는 "
                            "정상적으로 작동합니다."
                        )


                        st.warning(
                            "LLM 답변을 사용하려면 "
                            "Streamlit Secrets에 "
                            "OPENAI_API_KEY를 등록해주세요."
                        )


                        st.write(
                            "현재 검색된 자료를 확인할 수 있습니다."
                        )


                    else:

                        with st.spinner(
                            "관련 자료를 바탕으로 "
                            "답변을 작성하고 있습니다..."
                        ):

                            (
                                answer,
                                error
                            ) = generate_answer(
                                question,
                                good_results
                            )


                        if error:

                            st.error(
                                "AI 답변 생성 중 오류가 발생했습니다."
                            )

                            st.code(
                                error
                            )

                        else:

                            st.markdown(
                                answer
                            )


                            st.divider()


                            st.caption(
                                "📚 위 답변은 등록된 "
                                "통계조사 자료를 검색한 뒤 "
                                "생성되었습니다."
                            )


# ============================================================
# 자료관리
# ============================================================

elif menu == "자료관리":

    st.subheader(
        "📂 자료관리"
    )


    st.write(
        "PDF, TXT, 사진 자료를 등록하면 "
        "OCR/텍스트 추출 → Chunk → 의미 임베딩 → "
        "RAG 검색에 사용할 수 있습니다."
    )


    st.divider()


    # ========================================================
    # PDF
    # ========================================================

    st.markdown(
        "### 📄 PDF 자료"
    )


    pdf_files = st.file_uploader(
        "PDF 파일을 선택하세요.",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_upload"
    )


    # ========================================================
    # TXT
    # ========================================================

    st.markdown(
        "### 📝 TXT 자료"
    )


    txt_files = st.file_uploader(
        "TXT 파일을 선택하세요.",
        type=["txt"],
        accept_multiple_files=True,
        key="txt_upload"
    )


    # ========================================================
    # 사진
    # ========================================================

    st.markdown(
        "### 📷 사진 자료"
    )


    image_files = st.file_uploader(
        "사진 파일을 선택하세요.",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
            "heic",
            "heif",
            "bmp"
        ],
        accept_multiple_files=True,
        key="image_upload"
    )


    # ========================================================
    # PDF
    # ========================================================

    if pdf_files:

        st.divider()

        st.subheader(
            "📄 PDF 처리"
        )


        for index, file in enumerate(
            pdf_files
        ):

            try:

                file.seek(0)

                pdf_bytes = file.read()


                full_text, page_count = (
                    extract_pdf_text(
                        pdf_bytes
                    )
                )


                with st.expander(
                    f"📄 {file.name}",
                    expanded=True
                ):

                    st.write(
                        f"페이지 수: {page_count}"
                    )


                    if full_text:

                        st.text_area(
                            "추출된 텍스트",
                            full_text,
                            height=350,
                            key=(
                                "pdf_preview_"
                                + str(index)
                                + file.name
                            )
                        )


                        if st.button(
                            "💾 PDF 자료 등록",
                            key=(
                                "register_pdf_"
                                + str(index)
                                + file.name
                            ),
                            use_container_width=True
                        ):

                            success, message = (
                                register_document(
                                    file.name,
                                    "PDF",
                                    full_text,
                                    file.size,
                                    page_count
                                )
                            )


                            if success:

                                st.success(
                                    message
                                )

                                st.rerun()

                            else:

                                st.warning(
                                    message
                                )


                    else:

                        st.warning(
                            "PDF에서 텍스트를 찾지 못했습니다."
                        )


            except Exception as e:

                st.error(
                    f"{file.name} 처리 중 오류가 발생했습니다."
                )

                st.code(
                    repr(e)
                )


    # ========================================================
    # TXT
    # ========================================================

    if txt_files:

        st.divider()

        st.subheader(
            "📝 TXT 처리"
        )


        for index, file in enumerate(
            txt_files
        ):

            try:

                file.seek(0)

                content_bytes = file.read()


                try:

                    content = content_bytes.decode(
                        "utf-8"
                    )

                except UnicodeDecodeError:

                    content = content_bytes.decode(
                        "cp949"
                    )


                with st.expander(
                    f"📝 {file.name}",
                    expanded=True
                ):

                    st.text_area(
                        "TXT 내용",
                        content,
                        height=300,
                        key=(
                            "txt_preview_"
                            + str(index)
                            + file.name
                        )
                    )


                    if st.button(
                        "💾 TXT 자료 등록",
                        key=(
                            "register_txt_"
                            + str(index)
                            + file.name
                        ),
                        use_container_width=True
                    ):

                        success, message = (
                            register_document(
                                file.name,
                                "TXT",
                                content,
                                file.size
                            )
                        )


                        if success:

                            st.success(
                                message
                            )

                            st.rerun()

                        else:

                            st.warning(
                                message
                            )


            except Exception as e:

                st.error(
                    f"{file.name} 처리 중 오류가 발생했습니다."
                )

                st.code(
                    repr(e)
                )


    # ========================================================
    # 사진 OCR
    # ========================================================

    if image_files:

        st.divider()

        st.subheader(
            "📷 사진 OCR"
        )


        for index, file in enumerate(
            image_files
        ):

            with st.expander(
                f"📷 {file.name}",
                expanded=True
            ):

                try:

                    file.seek(0)

                    image_bytes = file.read()


                    if not PIL_SUPPORT:

                        st.error(
                            "Pillow가 설치되어 있지 않습니다."
                        )

                        continue


                    image = Image.open(
                        io.BytesIO(
                            image_bytes
                        )
                    )


                    image.load()


                    st.image(
                        image,
                        caption=file.name,
                        use_container_width=True
                    )


                    st.caption(
                        f"이미지 크기: "
                        f"{image.width} × {image.height}"
                    )


                    if st.button(
                        "🔎 한글 OCR 실행",
                        key=(
                            "ocr_button_"
                            + str(index)
                            + file.name
                        ),
                        use_container_width=True
                    ):

                        with st.spinner(
                            "OCR 모델을 준비하고 있습니다..."
                        ):

                            (
                                models_ok,
                                model_results
                            ) = prepare_ocr_models()


                        if not models_ok:

                            st.error(
                                "OCR 모델 준비에 실패했습니다."
                            )


                            for (
                                model_name,
                                success,
                                message
                            ) in model_results:

                                if success:

                                    st.success(
                                        model_name
                                        + ": "
                                        + message
                                    )

                                else:

                                    st.error(
                                        model_name
                                        + ": "
                                        + message
                                    )


                        else:

                            with st.spinner(
                                "사진에서 한글을 읽고 있습니다..."
                            ):

                                (
                                    result,
                                    error
                                ) = run_ocr(
                                    image
                                )


                            if error:

                                st.error(
                                    "OCR 실행 중 오류가 발생했습니다."
                                )

                                st.code(
                                    error
                                )

                            else:

                                texts = (
                                    extract_ocr_text(
                                        result
                                    )
                                )


                                if texts:

                                    ocr_text = (
                                        "\n".join(
                                            texts
                                        )
                                    )


                                    ocr_key = (
                                        "ocr_text_"
                                        + str(index)
                                        + file.name
                                    )


                                    st.session_state[
                                        ocr_key
                                    ] = ocr_text


                                    st.success(
                                        "한글 OCR이 완료되었습니다."
                                    )

                                else:

                                    st.warning(
                                        "인식된 텍스트가 없습니다."
                                    )


                    ocr_key = (
                        "ocr_text_"
                        + str(index)
                        + file.name
                    )


                    if ocr_key in st.session_state:

                        ocr_text = (
                            st.session_state[
                                ocr_key
                            ]
                        )


                        st.text_area(
                            "📝 OCR 결과",
                            ocr_text,
                            height=350,
                            key=(
                                "ocr_result_"
                                + str(index)
                                + file.name
                            )
                        )


                        if st.button(
                            "💾 OCR 결과를 자료로 등록",
                            key=(
                                "register_ocr_"
                                + str(index)
                                + file.name
                            ),
                            use_container_width=True
                        ):

                            success, message = (
                                register_document(
                                    file.name,
                                    "사진 OCR",
                                    ocr_text,
                                    file.size
                                )
                            )


                            if success:

                                st.success(
                                    message
                                )

                                st.rerun()

                            else:

                                st.warning(
                                    message
                                )


                except Exception as e:

                    st.error(
                        "이미지를 처리할 수 없습니다."
                    )

                    st.code(
                        repr(e)
                    )


    # ========================================================
    # 등록 자료
    # ========================================================

    st.divider()

    st.subheader(
        "📚 등록된 자료"
    )


    documents = load_documents()


    if not documents:

        st.info(
            "등록된 자료가 없습니다."
        )

    else:

        st.success(
            f"총 {len(documents)}개의 자료가 등록되어 있습니다."
        )


        for index, document in enumerate(
            reversed(documents)
        ):

            with st.expander(
                f"📚 {document.get('filename', '')}"
                f"  |  {document.get('type', '')}"
            ):

                st.write(
                    "**등록일:** "
                    + document.get(
                        "registered_at",
                        ""
                    )
                )


                st.write(
                    "**텍스트 길이:** "
                    + f"{document.get('text_length', 0):,}자"
                )


                if document.get(
                    "page_count"
                ):

                    st.write(
                        "**페이지:** "
                        + str(
                            document.get(
                                "page_count"
                            )
                        )
                    )


# ============================================================
# 하단 안내
# ============================================================

st.divider()

st.caption(
    "현재 구조: 업로드 → OCR/텍스트 추출 → Chunk → "
    "다국어 임베딩 → 의미 기반 검색 → LLM 답변"
)
