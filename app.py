import io
import os
import sys
import importlib.metadata

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
# 패키지 상태
# ============================================================

# Pillow
try:
    from PIL import Image
    PIL_SUPPORT = True
    PIL_ERROR = ""
except Exception as e:
    PIL_SUPPORT = False
    PIL_ERROR = repr(e)


# HEIC
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()

    HEIC_SUPPORT = True
    HEIC_ERROR = ""

except Exception as e:

    HEIC_SUPPORT = False
    HEIC_ERROR = repr(e)


# RapidOCR
try:

    from rapidocr import RapidOCR

    OCR_SUPPORT = True
    OCR_IMPORT_ERROR = ""

except Exception as e:

    OCR_SUPPORT = False
    OCR_IMPORT_ERROR = repr(e)


# Requests
try:

    import requests

    REQUESTS_SUPPORT = True
    REQUESTS_ERROR = ""

except Exception as e:

    REQUESTS_SUPPORT = False
    REQUESTS_ERROR = repr(e)


# NumPy
try:

    import numpy as np

    NUMPY_SUPPORT = True
    NUMPY_ERROR = ""

except Exception as e:

    NUMPY_SUPPORT = False
    NUMPY_ERROR = repr(e)


# ============================================================
# 버전
# ============================================================

try:
    RAPIDOCR_VERSION = importlib.metadata.version("rapidocr")
except Exception:
    RAPIDOCR_VERSION = "확인 불가"


try:
    ONNXRUNTIME_VERSION = importlib.metadata.version(
        "onnxruntime"
    )
except Exception:
    ONNXRUNTIME_VERSION = "확인 불가"


# ============================================================
# 모델 저장 폴더
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# RapidOCR 3.9.2 공식 모델
#
# /tmp/rapidocr_models를 사용하지 않고
# 우리 앱의 models 폴더에 직접 저장
# ============================================================

MODEL_URLS = {

    "det": (
        "https://www.modelscope.cn/models/"
        "RapidAI/RapidOCR/resolve/v3.9.2/"
        "onnx/PP-OCRv5/det/"
        "ch_PP-OCRv5_det_mobile.onnx"
    ),

    "cls": (
        "https://www.modelscope.cn/models/"
        "RapidAI/RapidOCR/resolve/v3.9.2/"
        "onnx/PP-OCRv5/cls/"
        "ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx"
    ),

    "rec": (
        "https://www.modelscope.cn/models/"
        "RapidAI/RapidOCR/resolve/v3.9.2/"
        "onnx/PP-OCRv5/rec/"
        "korean_PP-OCRv5_rec_mobile.onnx"
    ),

    "dict": (
        "https://www.modelscope.cn/models/"
        "RapidAI/RapidOCR/resolve/v3.9.2/"
        "paddle/PP-OCRv5/rec/"
        "korean_PP-OCRv5_rec_mobile/"
        "ppocrv5_korean_dict.txt"
    )
}


MODEL_FILES = {

    "det": os.path.join(
        MODEL_DIR,
        "ch_PP-OCRv5_det_mobile.onnx"
    ),

    "cls": os.path.join(
        MODEL_DIR,
        "ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx"
    ),

    "rec": os.path.join(
        MODEL_DIR,
        "korean_PP-OCRv5_rec_mobile.onnx"
    ),

    "dict": os.path.join(
        MODEL_DIR,
        "ppocrv5_korean_dict.txt"
    )
}


# ============================================================
# 모델 다운로드
# ============================================================

def download_file(
    url,
    destination
):

    # 이미 파일이 존재하면 다운로드하지 않음
    if os.path.exists(destination):

        size = os.path.getsize(
            destination
        )

        if size > 0:

            return True, (
                f"이미 존재: "
                f"{os.path.basename(destination)}"
            )


    if not REQUESTS_SUPPORT:

        return False, (
            "requests 패키지를 사용할 수 없습니다."
        )


    try:

        response = requests.get(
            url,
            stream=True,
            timeout=120
        )

        response.raise_for_status()


        with open(
            destination,
            "wb"
        ) as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:

                    f.write(chunk)


        size = os.path.getsize(
            destination
        )


        if size == 0:

            try:
                os.remove(destination)
            except Exception:
                pass

            return False, (
                "다운로드된 파일 크기가 0입니다."
            )


        return True, (
            f"다운로드 완료: "
            f"{os.path.basename(destination)}"
        )


    except Exception as e:

        try:

            if os.path.exists(
                destination
            ):

                os.remove(
                    destination
                )

        except Exception:
            pass


        return False, repr(e)


# ============================================================
# 모든 OCR 모델 준비
# ============================================================

@st.cache_resource
def prepare_ocr_models():

    results = []

    all_success = True


    for key in [
        "det",
        "cls",
        "rec",
        "dict"
    ]:

        success, message = download_file(
            MODEL_URLS[key],
            MODEL_FILES[key]
        )

        results.append(
            (
                key,
                success,
                message
            )
        )

        if not success:

            all_success = False


    return all_success, results


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


    # --------------------------------------------------------
    # 중요:
    # 기본 /tmp/rapidocr_models를 사용하지 않음
    # --------------------------------------------------------

    try:

        engine = RapidOCR(

            params={

                # --------------------------------------------
                # Detection
                # --------------------------------------------

                "Det.engine_type": "onnxruntime",

                "Det.lang_type": "ch",

                "Det.model_type": "mobile",

                "Det.ocr_version": "PP-OCRv5",

                "Det.model_path":
                    MODEL_FILES["det"],


                # --------------------------------------------
                # Classification
                # --------------------------------------------

                "Cls.engine_type": "onnxruntime",

                "Cls.model_path":
                    MODEL_FILES["cls"],


                # --------------------------------------------
                # Recognition
                # --------------------------------------------

                "Rec.engine_type": "onnxruntime",

                "Rec.lang_type": "korean",

                "Rec.model_type": "mobile",

                "Rec.ocr_version": "PP-OCRv5",

                "Rec.model_path":
                    MODEL_FILES["rec"],

                "Rec.rec_keys_path":
                    MODEL_FILES["dict"]

            }

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

    if not OCR_SUPPORT:

        return None, (
            "RapidOCR을 불러오지 못했습니다.\n"
            + OCR_IMPORT_ERROR
        )


    if not NUMPY_SUPPORT:

        return None, (
            "NumPy를 불러오지 못했습니다.\n"
            + NUMPY_ERROR
        )


    try:

        image = pil_image.convert(
            "RGB"
        )

        image_array = np.array(
            image
        )


        engine = create_ocr_engine()


        result = engine(
            image_array
        )


        return result, None


    except Exception as e:

        return None, repr(e)


# ============================================================
# OCR 결과에서 텍스트 추출
# ============================================================

def extract_ocr_text(
    result
):

    texts = []


    try:

        # ----------------------------------------------------
        # 최신 RapidOCR 결과
        # ----------------------------------------------------

        if hasattr(
            result,
            "txts"
        ):

            values = result.txts

            if values is not None:

                for value in values:

                    if value is not None:

                        text = str(
                            value
                        ).strip()

                        if text:

                            texts.append(
                                text
                            )

            return texts


        # ----------------------------------------------------
        # tuple 결과
        # ----------------------------------------------------

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

                            if value is not None:

                                text = str(
                                    value
                                ).strip()

                                if text:

                                    texts.append(
                                        text
                                    )

                        return texts


        # ----------------------------------------------------
        # list 결과
        # ----------------------------------------------------

        if isinstance(
            result,
            list
        ):

            for item in result:

                if isinstance(
                    item,
                    str
                ):

                    text = item.strip()

                    if text:

                        texts.append(
                            text
                        )


    except Exception:

        pass


    return texts


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


    st.write(
        "**OCR 시스템 상태**"
    )


    if OCR_SUPPORT:

        st.success(
            "✅ RapidOCR 설치됨"
        )

    else:

        st.error(
            "❌ RapidOCR 설치 오류"
        )


    st.write(
        f"RapidOCR: {RAPIDOCR_VERSION}"
    )

    st.write(
        f"ONNX Runtime: {ONNXRUNTIME_VERSION}"
    )


    st.divider()


    with st.expander(
        "🔧 시스템 진단"
    ):

        st.write(
            "Python"
        )

        st.code(
            sys.version
        )


        st.write(
            "RapidOCR"
        )

        st.code(
            RAPIDOCR_VERSION
        )


        st.write(
            "ONNX Runtime"
        )

        st.code(
            ONNXRUNTIME_VERSION
        )


        st.write(
            "모델 폴더"
        )

        st.code(
            MODEL_DIR
        )


        st.write(
            "RapidOCR import"
        )

        if OCR_SUPPORT:

            st.success(
                "성공"
            )

        else:

            st.error(
                "실패"
            )

            st.code(
                OCR_IMPORT_ERROR
            )


# ============================================================
# 질문하기
# ============================================================

if menu == "질문하기":

    st.subheader(
        "💬 질문하기"
    )


    question = st.text_area(
        "궁금한 내용을 입력하세요.",
        placeholder=(
            "예: 취업자는 어떤 기준으로 판단하나요?"
        ),
        height=120
    )


    if st.button(
        "🔍 질문하기",
        use_container_width=True
    ):

        if question.strip():

            st.info(
                "현재는 문서 처리 단계입니다. "
                "추후 AI가 등록된 자료를 검색하여 "
                "답변하도록 연결합니다."
            )


            st.write(
                "입력하신 질문:"
            )

            st.write(
                question
            )


        else:

            st.warning(
                "질문을 입력해주세요."
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
        "등록하고 처리할 수 있습니다."
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
        "JPG, JPEG, PNG, WEBP, HEIC, HEIF 등 "
        "이미지 파일을 선택할 수 있습니다."
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


    pdf_count = (
        len(pdf_files)
        if pdf_files
        else 0
    )


    txt_count = (
        len(txt_files)
        if txt_files
        else 0
    )


    image_count = (
        len(image_files)
        if image_files
        else 0
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "📄 PDF",
            pdf_count
        )


    with col2:

        st.metric(
            "📝 TXT",
            txt_count
        )


    with col3:

        st.metric(
            "📷 사진",
            image_count
        )


    # ========================================================
    # PDF 내용 확인
    # ========================================================

    if pdf_files:

        st.divider()


        st.subheader(
            "📄 PDF 내용 확인"
        )


        for file in pdf_files:

            try:

                file.seek(0)

                pdf_bytes = file.read()


                reader = PdfReader(
                    io.BytesIO(
                        pdf_bytes
                    )
                )


                full_text = ""


                for page_number, page in enumerate(
                    reader.pages,
                    start=1
                ):

                    text = page.extract_text()


                    if text:

                        full_text += (
                            "\n\n"
                            f"===== 페이지 {page_number} ====="
                            "\n\n"
                        )


                        full_text += text


                with st.expander(
                    f"📄 {file.name}"
                ):

                    if full_text.strip():

                        st.text_area(
                            "추출된 텍스트",
                            full_text,
                            height=500,
                            key=f"pdf_text_{file.name}"
                        )


                        st.success(
                            f"{len(reader.pages)}페이지에서 "
                            "텍스트를 추출했습니다."
                        )


                    else:

                        st.warning(
                            "PDF에서 텍스트를 찾지 못했습니다."
                        )


                        st.info(
                            "스캔 PDF일 가능성이 있습니다."
                        )


            except Exception as e:

                st.error(
                    f"{file.name} 처리 중 오류가 발생했습니다."
                )


                st.code(
                    repr(e)
                )


    # ========================================================
    # TXT 내용 확인
    # ========================================================

    if txt_files:

        st.divider()


        st.subheader(
            "📝 TXT 내용 확인"
        )


        for file in txt_files:

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
                    f"📝 {file.name}"
                ):

                    st.text_area(
                        "TXT 내용",
                        content,
                        height=400,
                        key=f"txt_text_{file.name}"
                    )


                    st.success(
                        "TXT 파일을 정상적으로 읽었습니다."
                    )


            except Exception as e:

                st.error(
                    f"{file.name} 파일을 읽을 수 없습니다."
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
            "📷 업로드된 사진"
        )


        for index, file in enumerate(
            image_files
        ):

            with st.expander(
                f"📷 {file.name}",
                expanded=True
            ):

                # --------------------------------------------
                # 파일 정보
                # --------------------------------------------

                col_a, col_b, col_c = st.columns(3)


                with col_a:

                    st.write(
                        "**파일명**"
                    )

                    st.code(
                        file.name
                    )


                with col_b:

                    st.write(
                        "**MIME 타입**"
                    )

                    st.code(
                        str(file.type)
                    )


                with col_c:

                    st.write(
                        "**크기**"
                    )

                    st.code(
                        f"{file.size / 1024 / 1024:.2f} MB"
                    )


                # --------------------------------------------
                # 이미지 열기
                # --------------------------------------------

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


                    st.write(
                        "**실제 이미지 포맷**"
                    )

                    st.code(
                        str(image.format)
                    )


                    st.image(
                        image,
                        caption=file.name,
                        use_container_width=True
                    )


                    st.success(
                        "사진을 정상적으로 읽었습니다."
                    )


                    # ========================================
                    # OCR
                    # ========================================

                    st.markdown(
                        "### 🔎 한국어 OCR"
                    )


                    if not OCR_SUPPORT:

                        st.error(
                            "RapidOCR을 사용할 수 없습니다."
                        )

                        st.code(
                            OCR_IMPORT_ERROR
                        )


                    else:

                        if st.button(
                            "🔎 이 사진 OCR 실행",
                            key=f"ocr_button_{index}_{file.name}",
                            use_container_width=True
                        ):

                            # --------------------------------
                            # 모델 준비
                            # --------------------------------

                            with st.spinner(
                                "한국어 OCR 모델을 준비하고 있습니다..."
                            ):

                                models_ok, model_results = (
                                    prepare_ocr_models()
                                )


                            # 모델 상태 표시

                            with st.expander(
                                "📦 OCR 모델 상태",
                                expanded=False
                            ):

                                for (
                                    model_name,
                                    success,
                                    message
                                ) in model_results:

                                    if success:

                                        st.success(
                                            f"{model_name}: "
                                            f"{message}"
                                        )

                                    else:

                                        st.error(
                                            f"{model_name}: "
                                            f"{message}"
                                        )


                            if not models_ok:

                                st.error(
                                    "OCR 모델 준비에 실패했습니다."
                                )

                                st.info(
                                    "위의 OCR 모델 상태에서 "
                                    "어떤 모델 다운로드가 실패했는지 확인해주세요."
                                )

                                continue


                            # --------------------------------
                            # OCR 엔진
                            # --------------------------------

                            with st.spinner(
                                "사진의 글자를 인식하고 있습니다..."
                            ):

                                result, error = run_ocr(
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

                                texts = extract_ocr_text(
                                    result
                                )


                                if texts:

                                    ocr_text = "\n".join(
                                        texts
                                    )


                                    st.success(
                                        f"{len(texts)}개의 "
                                        "텍스트 영역을 인식했습니다."
                                    )


                                    st.text_area(
                                        "📝 OCR 인식 결과",
                                        ocr_text,
                                        height=400,
                                        key=(
                                            f"ocr_result_"
                                            f"{index}_"
                                            f"{file.name}"
                                        )
                                    )


                                else:

                                    st.warning(
                                        "OCR은 실행됐지만 "
                                        "인식된 텍스트가 없습니다."
                                    )


                except Exception as e:

                    st.error(
                        "이미지를 처리할 수 없습니다."
                    )


                    st.code(
                        repr(e)
    )
