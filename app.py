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


# ============================================================
# 기본 폴더
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
# HEIC / HEIF
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


# ============================================================
# OCR 모델 폴더
# ============================================================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# OCR 모델 파일
# ============================================================

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
# 자료 목록 불러오기
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
# 자료 ID
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
        return False, "저장할 내용이 없습니다."


    text = text.strip()


    if not text:
        return False, "저장할 내용이 없습니다."


    documents = load_documents()


    text_bytes = text.encode(
        "utf-8"
    )


    file_id = make_file_id(
        filename,
        text_bytes
    )


    # 중복 확인
    for document in documents:

        if document.get("id") == file_id:

            return (
                False,
                "이미 등록된 자료입니다."
            )


    # 파일명에서 확장자 제거
    base_name = os.path.splitext(
        filename
    )[0]


    # 파일명에 사용할 수 없는 문자 일부 정리
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


    # 실제 텍스트 저장
    with open(
        text_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            text
        )


    document = {

        "id": file_id,

        "filename": filename,

        "type": doc_type,

        "text_file": text_filename,

        "size": original_size,

        "page_count": page_count,

        "text_length": len(text),

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
# 파일 다운로드
# ============================================================

def download_model(
    url,
    path
):

    # 이미 존재하면 사용
    if os.path.exists(path):

        try:

            if os.path.getsize(path) > 0:

                return True, "이미 존재"


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


        if not os.path.exists(
            path
        ):

            return (
                False,
                "다운로드 파일이 생성되지 않았습니다."
            )


        size = os.path.getsize(
            path
        )


        if size == 0:

            return (
                False,
                "다운로드된 파일 크기가 0입니다."
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


    files = [

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
    ]


    all_ok = True


    for name, url, path in files:

        success, message = download_model(
            url,
            path
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
# OCR 엔진 생성
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


    # 모델 파일 확인
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


        if os.path.getsize(
            path
        ) <= 0:

            raise FileNotFoundError(
                "OCR 모델 파일이 비어 있습니다:\n"
                + path
            )


    try:

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


        engine = RapidOCR(
            params=params
        )


        return engine


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

    if not PIL_SUPPORT:

        return (
            None,
            "Pillow가 설치되어 있지 않습니다."
        )


    if not NUMPY_SUPPORT:

        return (
            None,
            NUMPY_ERROR
        )


    try:

        # RGB로 변환
        image = pil_image.convert(
            "RGB"
        )


        # 너무 큰 사진은 OCR용으로 축소
        max_dimension = 2500


        width, height = image.size


        if max(
            width,
            height
        ) > max_dimension:

            ratio = (
                max_dimension
                / max(width, height)
            )


            new_size = (
                int(width * ratio),
                int(height * ratio)
            )


            image = image.resize(
                new_size
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
# OCR 결과에서 텍스트 추출
# ============================================================

def extract_ocr_text(
    result
):

    texts = []


    try:

        # RapidOCR 최신 결과
        if hasattr(
            result,
            "txts"
        ):

            values = result.txts


            if values is not None:

                for value in values:

                    if value is None:
                        continue


                    text = str(
                        value
                    ).strip()


                    if text:

                        texts.append(
                            text
                        )


            return texts


        # tuple 형태 결과
        if isinstance(
            result,
            tuple
        ):

            for item in result:

                if hasattr(
                    item,
                    "txts"
                ):

                    values = item.txts


                    if values is not None:

                        for value in values:

                            if value is None:
                                continue


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
# 검색용 텍스트 정리
# ============================================================

def normalize_text(
    text
):

    return (
        text
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
    )


# ============================================================
# 검색
# ============================================================

def search_documents(
    question,
    documents,
    max_results=5
):

    question = question.strip()


    if not question:

        return []


    # 질문을 공백 기준으로 나눔
    raw_words = question.split()


    # 너무 짧은 단어 제외
    query_words = []


    for word in raw_words:

        word = word.strip(
            ".,?!:;()[]{}\"'"
        )


        if len(word) >= 2:

            query_words.append(
                word
            )


    # 질문 자체도 검색어로 사용
    if not query_words:

        query_words = [
            question
        ]


    results = []


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


        if not text.strip():
            continue


        # --------------------------------------------------------
        # 문장/구간 분리
        # --------------------------------------------------------

        normalized = normalize_text(
            text
        )


        # 한국어 문장 구분
        pieces = []


        current = ""


        for char in normalized:

            current += char


            if char in [
                ".",
                "。",
                "?",
                "!",
                ";"
            ]:

                if current.strip():

                    pieces.append(
                        current.strip()
                    )

                current = ""


        if current.strip():

            pieces.append(
                current.strip()
            )


        # 문장이 너무 길면 일정 길이로 분리
        final_pieces = []


        for piece in pieces:

            if len(piece) <= 500:

                final_pieces.append(
                    piece
                )

            else:

                for i in range(
                    0,
                    len(piece),
                    400
                ):

                    part = piece[
                        i:i + 400
                    ].strip()


                    if part:

                        final_pieces.append(
                            part
                        )


        # --------------------------------------------------------
        # 점수 계산
        # --------------------------------------------------------

        matched = []


        for piece in final_pieces:

            lower_piece = piece.lower()

            score = 0


            for word in query_words:

                if word.lower() in lower_piece:

                    score += 1


                    # 정확히 많이 등장하면 추가 점수
                    count = lower_piece.count(
                        word.lower()
                    )


                    if count > 1:

                        score += min(
                            count - 1,
                            2
                        )


            if score > 0:

                matched.append(
                    (
                        score,
                        piece
                    )
                )


        if not matched:

            continue


        matched.sort(
            reverse=True,
            key=lambda x: x[0]
        )


        best_sentences = [
            item[1]
            for item in matched[:5]
        ]


        total_score = sum(
            item[0]
            for item in matched
        )


        results.append(
            {

                "score":
                    total_score,

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

                "sentences":
                    best_sentences,

                "document":
                    document
            }
        )


    # 점수가 높은 자료부터
    results.sort(
        reverse=True,
        key=lambda x: x["score"]
    )


    return results[
        :max_results
    ]


# ============================================================
# 제목
# ============================================================

st.title(
    "📚 가구부문통계조사 챗봇"
)


st.caption(
    "가구부문 통계조사 업무자료를 기반으로 "
    "질문에 답변하는 AI 챗봇"
)


st.divider()


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
            "✅ RapidOCR 사용 가능"
        )

    else:

        st.error(
            "❌ RapidOCR 사용 불가"
        )


    st.caption(
        f"RapidOCR: {RAPIDOCR_VERSION}"
    )


    st.caption(
        f"ONNX Runtime: {ONNXRUNTIME_VERSION}"
    )


    if HEIC_SUPPORT:

        st.caption(
            "HEIC/HEIF: ✅"
        )

    else:

        st.caption(
            "HEIC/HEIF: ❌"
        )


    st.divider()


    documents = load_documents()


    st.metric(
        "📚 등록된 자료",
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
            "아직 등록된 자료가 없습니다."
        )

        st.write(
            "자료관리에서 PDF, TXT 또는 사진을 "
            "먼저 등록해주세요."
        )


    else:

        st.success(
            f"현재 {len(documents)}개의 자료가 "
            "검색 대상입니다."
        )


        question = st.text_area(
            "궁금한 내용을 입력하세요.",
            placeholder=(
                "예: 취업자는 어떤 기준으로 판단하나요?"
            ),
            height=130,
            key="question_input"
        )


        if st.button(
            "🔍 자료에서 검색",
            use_container_width=True
        ):

            if not question.strip():

                st.warning(
                    "질문을 입력해주세요."
                )

            else:

                with st.spinner(
                    "등록된 자료를 검색하고 있습니다..."
                ):

                    results = search_documents(
                        question,
                        documents,
                        max_results=5
                    )


                st.session_state[
                    "search_results"
                ] = results


                st.session_state[
                    "last_question"
                ] = question


        # --------------------------------------------------------
        # 검색 결과 표시
        # --------------------------------------------------------

        if (
            "search_results"
            in st.session_state
        ):

            results = st.session_state[
                "search_results"
            ]


            st.divider()


            st.subheader(
                "🔎 검색 결과"
            )


            if not results:

                st.warning(
                    "질문과 관련된 내용을 "
                    "등록된 자료에서 찾지 못했습니다."
                )


                st.info(
                    "질문의 표현을 조금 바꿔서 "
                    "다시 검색해보세요."
                )


            else:

                st.success(
                    f"{len(results)}개의 자료에서 "
                    "관련 내용을 찾았습니다."
                )


                for number, result in enumerate(
                    results,
                    start=1
                ):

                    filename = result[
                        "filename"
                    ]


                    doc_type = result[
                        "type"
                    ]


                    score = result[
                        "score"
                    ]


                    with st.expander(
                        f"📚 {number}. "
                        f"{filename} "
                        f"({doc_type})",
                        expanded=True
                    ):

                        st.caption(
                            f"검색 관련도 점수: {score}"
                        )


                        st.markdown(
                            "### 📖 관련 내용"
                        )


                        for sentence in result[
                            "sentences"
                        ]:

                            st.markdown(
                                "> "
                                + sentence
                            )


                st.divider()


                st.info(
                    "✅ 자료 검색이 완료되었습니다. "
                    "현재는 검색 결과를 보여주는 단계입니다. "
                    "다음 단계에서 LLM을 연결하면 "
                    "이 내용을 바탕으로 자연어 답변을 생성할 수 있습니다."
                )


# ============================================================
# 자료관리
# ============================================================

elif menu == "자료관리":

    st.subheader(
        "📂 자료관리"
    )


    st.write(
        "가구부문 통계조사 관련 PDF, TXT, 사진 자료를 "
        "등록할 수 있습니다."
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
        type=[
            "pdf"
        ],
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
        type=[
            "txt"
        ],
        accept_multiple_files=True,
        key="txt_upload"
    )


    # ========================================================
    # 사진
    # ========================================================

    st.markdown(
        "### 📷 사진 자료"
    )


    st.write(
        "JPG, JPEG, PNG, WEBP, HEIC, HEIF, BMP를 지원합니다."
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
    # 업로드 현황
    # ========================================================

    st.divider()


    st.subheader(
        "📊 업로드 현황"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "📄 PDF",
            len(pdf_files)
            if pdf_files
            else 0
        )


    with col2:

        st.metric(
            "📝 TXT",
            len(txt_files)
            if txt_files
            else 0
        )


    with col3:

        st.metric(
            "📷 사진",
            len(image_files)
            if image_files
            else 0
        )


    # ========================================================
    # PDF 처리
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
                            "💾 이 PDF 자료 등록",
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


                        st.info(
                            "스캔 PDF라면 다음 단계에서 "
                            "PDF OCR 기능을 추가할 수 있습니다."
                        )


            except Exception as e:

                st.error(
                    f"{file.name} 처리 중 오류가 발생했습니다."
                )


                st.code(
                    repr(e)
                )


    # ========================================================
    # TXT 처리
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
                        "💾 이 TXT 자료 등록",
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
    # 사진 처리 + OCR
    # ========================================================

    if image_files:

        st.divider()


        st.subheader(
            "📷 사진 OCR 및 자료 등록"
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
                            "Pillow가 설치되지 않았습니다."
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


                    col_a, col_b = st.columns(2)


                    with col_a:

                        st.write(
                            "**실제 이미지 크기**"
                        )

                        st.code(
                            f"{image.width} × "
                            f"{image.height}"
                        )


                    with col_b:

                        st.write(
                            "**파일 크기**"
                        )

                        st.code(
                            f"{file.size / 1024 / 1024:.2f} MB"
                        )


                    # --------------------------------------------
                    # OCR 실행
                    # --------------------------------------------

                    if st.button(
                        "🔎 이 사진 OCR 실행",
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
                                        f"{len(texts)}개 영역의 "
                                        "텍스트를 인식했습니다."
                                    )


                                else:

                                    st.warning(
                                        "인식된 텍스트가 없습니다."
                                    )


                    # --------------------------------------------
                    # OCR 결과 표시
                    # --------------------------------------------

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
                                "ocr_result_view_"
                                + str(index)
                                + file.name
                            )
                        )


                        st.divider()


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
    # 등록된 자료
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
            f"현재 총 {len(documents)}개의 자료가 등록되어 있습니다."
        )


        for index, document in enumerate(
            reversed(documents)
        ):

            filename = document.get(
                "filename",
                ""
            )


            doc_type = document.get(
                "type",
                ""
            )


            with st.expander(
                f"📚 {filename}  |  {doc_type}"
            ):

                col1, col2 = st.columns(2)


                with col1:

                    st.write(
                        "**자료명**"
                    )

                    st.write(
                        filename
                    )


                    st.write(
                        "**자료 유형**"
                    )

                    st.write(
                        doc_type
                    )


                    st.write(
                        "**등록일**"
                    )

                    st.write(
                        document.get(
                            "registered_at",
                            ""
                        )
                    )


                with col2:

                    st.write(
                        "**텍스트 길이**"
                    )

                    st.write(
                        f"{document.get('text_length', 0):,}자"
                    )


                    if document.get(
                        "page_count"
                    ):

                        st.write(
                            "**페이지 수**"
                        )

                        st.write(
                            document.get(
                                "page_count"
                            )
                        )


                # 저장된 텍스트 확인
                text_filename = document.get(
                    "text_file"
                )


                if text_filename:

                    text_path = os.path.join(
                        DATA_DIR,
                        text_filename
                    )


                    if os.path.exists(
                        text_path
                    ):

                        try:

                            with open(
                                text_path,
                                "r",
                                encoding="utf-8"
                            ) as f:

                                saved_text = f.read()


                            st.text_area(
                                "저장된 내용",
                                saved_text,
                                height=250,
                                key=(
                                    "saved_"
                                    + str(index)
                                    + "_"
                                    + document.get(
                                        "id",
                                        ""
                                    )
                                )
                            )


                        except Exception as e:

                            st.warning(
                                "저장된 자료를 읽을 수 없습니다."
                            )

                            st.code(
                                repr(e)
        )
