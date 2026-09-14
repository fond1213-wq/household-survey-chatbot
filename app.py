import io
import os
import sys
import importlib.metadata

import streamlit as st
from pypdf import PdfReader

# ============================================================
# HEIC / HEIF 지원
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
# RapidOCR 지원 확인 및 Enum 임포트
# ============================================================

OCR_SUPPORT = False
OCR_ERROR = ""
RAPIDOCR_VERSION = "확인 불가"
ONNXRUNTIME_VERSION = "확인 불가"

try:
    RAPIDOCR_VERSION = importlib.metadata.version("rapidocr")
except Exception:
    pass

try:
    ONNXRUNTIME_VERSION = importlib.metadata.version("onnxruntime")
except Exception:
    pass

try:
    from rapidocr import (
        RapidOCR,
        LangDet,
        LangRec,
        OCRVersion,
        ModelType,
        EngineType,
    )
    OCR_SUPPORT = True
except Exception as e:
    OCR_SUPPORT = False
    OCR_ERROR = repr(e)


# ============================================================
# 모델 캐시 경로 설정 (Streamlit Cloud 쓰기 권한 우회)
# ============================================================

# Streamlit Cloud의 /tmp는 쓰기 가능
MODEL_CACHE_DIR = "/tmp/rapidocr_models"
try:
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
except Exception:
    MODEL_CACHE_DIR = os.path.abspath("./rapidocr_models")
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


# ============================================================
# Streamlit 설정
# ============================================================

st.set_page_config(
    page_title="가구부문통계조사 챗봇",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# OCR 엔진 (한국어 특화 + 다국어 폴백 + 로컬 캐시 경로)
# ============================================================

@st.cache_resource
def get_ocr_engine():
    """
    한국어 인식에 최적화된 RapidOCR 엔진을 반환합니다.
    - 감지(Det): 다국어 모델 (LangDet.MULTI)
    - 인식(Rec): 한국어 모델 (LangRec.KOREAN)
    - 모델 캐시: /tmp/rapidocr_models (쓰기 가능 경로)
    실패 시 기본 중문 모델로 자동 대체합니다.
    """
    if not OCR_SUPPORT:
        raise RuntimeError(OCR_ERROR)

    # --------------------------------------------------------
    # 1차 시도: 한국어 특화 설정 (Multi Det + Korean Rec)
    # --------------------------------------------------------
    try:
        engine = RapidOCR(
            params={
                # 감지: 다국어 (한국어 텍스트 영역 감지)
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Det.lang_type": LangDet.MULTI,
                "Det.model_type": ModelType.MOBILE,
                "Det.ocr_version": OCRVersion.PPOCRV5,
                # 감지 모델을 /tmp 캐시에 저장하도록 경로 지정
                # (버전에 따라 무시될 수 있으나, 있어도 문제 없음)
                "Det.model_path": os.path.join(
                    MODEL_CACHE_DIR,
                    "ch_PP-OCRv5_det_mobile.onnx"
                ),

                # 인식: 한국어
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.lang_type": LangRec.KOREAN,
                "Rec.model_type": ModelType.MOBILE,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.model_path": os.path.join(
                    MODEL_CACHE_DIR,
                    "korean_PP-OCRv5_rec_mobile.onnx"
                ),

                # 텍스트 방향 분류
                "Cls.engine_type": EngineType.ONNXRUNTIME,
                "Cls.lang_type": LangDet.CH,
                "Cls.model_type": ModelType.MOBILE,
                "Cls.ocr_version": OCRVersion.PPOCRV4,
                "Cls.model_path": os.path.join(
                    MODEL_CACHE_DIR,
                    "ch_ppocr_mobile_v2.0_cls_infer.onnx"
                ),
            }
        )
        return engine

    except Exception as e:
        # ----------------------------------------------------
        # 2차 시도: 한국어 모델 실패 시 기본 중문 모델
        # ----------------------------------------------------
        try:
            st.warning(
                f"한국어 모델 로드 실패. 기본 모델로 대체합니다. (원인: {e})"
            )
            engine = RapidOCR(
                params={
                    "Det.engine_type": EngineType.ONNXRUNTIME,
                    "Det.lang_type": LangDet.CH,
                    "Det.model_type": ModelType.MOBILE,
                    "Det.ocr_version": OCRVersion.PPOCRV5,
                    "Det.model_path": os.path.join(
                        MODEL_CACHE_DIR,
                        "ch_PP-OCRv5_det_mobile.onnx"
                    ),

                    "Rec.engine_type": EngineType.ONNXRUNTIME,
                    "Rec.lang_type": LangRec.CH,
                    "Rec.model_type": ModelType.MOBILE,
                    "Rec.ocr_version": OCRVersion.PPOCRV5,
                    "Rec.model_path": os.path.join(
                        MODEL_CACHE_DIR,
                        "ch_PP-OCRv5_rec_mobile.onnx"
                    ),

                    "Cls.engine_type": EngineType.ONNXRUNTIME,
                    "Cls.lang_type": LangDet.CH,
                    "Cls.model_type": ModelType.MOBILE,
                    "Cls.ocr_version": OCRVersion.PPOCRV4,
                    "Cls.model_path": os.path.join(
                        MODEL_CACHE_DIR,
                        "ch_ppocr_mobile_v2.0_cls_infer.onnx"
                    ),
                }
            )
            return engine

        except Exception as e2:
            # 최종 실패
            raise RuntimeError(f"OCR 엔진 초기화 실패: {repr(e2)}")


# ============================================================
# OCR 실행
# ============================================================

def run_ocr(pil_image):
    try:
        import numpy as np
        image_array = np.array(pil_image.convert("RGB"))
        engine = get_ocr_engine()
        result = engine(image_array)
        return result, None
    except Exception as e:
        return None, repr(e)


# ============================================================
# OCR 텍스트 추출
# ============================================================

def extract_ocr_text(result):
    texts = []
    try:
        if hasattr(result, "txts"):
            txts = result.txts
            if txts is not None:
                for text in txts:
                    if text is not None:
                        text = str(text).strip()
                        if text:
                            texts.append(text)
            return texts

        if isinstance(result, tuple):
            for item in result:
                if hasattr(item, "txts"):
                    txts = item.txts
                    if txts is not None:
                        for text in txts:
                            if text is not None:
                                text = str(text).strip()
                                if text:
                                    texts.append(text)
                        return texts

        if isinstance(result, list):
            for item in result:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        texts.append(text)
                elif isinstance(item, (list, tuple)):
                    for value in item:
                        if isinstance(value, str):
                            text = value.strip()
                            if text:
                                texts.append(text)
    except Exception:
        pass
    return texts


# ============================================================
# 제목
# ============================================================

st.title("📚 가구부문통계조사 챗봇")
st.caption(
    "가구부문 통계조사 업무자료를 기반으로 "
    "질문에 답변하는 AI 챗봇"
)
st.divider()


# ============================================================
# 사이드바
# ============================================================

with st.sidebar:
    st.header("📂 메뉴")
    menu = st.radio(
        "이동",
        ["질문하기", "자료관리"]
    )
    st.divider()

    st.write("**HEIC 지원**")
    if HEIC_SUPPORT:
        st.success("✅ 지원")
    else:
        st.warning("❌ 지원 안 됨")

    st.write("**OCR 지원**")
    if OCR_SUPPORT:
        st.success("✅ RapidOCR v3 지원")
        st.caption("한국어 인식 모델(korean) 로드 시도")
    else:
        st.error("❌ RapidOCR 지원 안 됨")

    st.divider()

    with st.expander("🔧 시스템 진단"):
        st.write("Python 버전")
        st.code(sys.version)

        st.write("RapidOCR 버전")
        st.code(RAPIDOCR_VERSION)

        st.write("ONNX Runtime 버전")
        st.code(ONNXRUNTIME_VERSION)

        st.write("모델 캐시 경로")
        st.code(MODEL_CACHE_DIR)

        if OCR_SUPPORT:
            st.success("RapidOCR import 성공")
        else:
            st.error("RapidOCR import 실패")
            st.code(OCR_ERROR)


# ============================================================
# 질문하기
# ============================================================

if menu == "질문하기":
    st.subheader("💬 질문하기")
    question = st.text_area(
        "궁금한 내용을 입력하세요.",
        placeholder="예: 취업자는 어떤 기준으로 판단하나요?",
        height=120
    )

    if st.button("🔍 질문하기", use_container_width=True):
        if question.strip():
            st.info(
                "현재는 문서 처리 단계입니다. "
                "추후 AI가 등록된 자료를 검색하여 답변하도록 연결합니다."
            )
            st.write("입력하신 질문:")
            st.write(question)
        else:
            st.warning("질문을 입력해주세요.")


# ============================================================
# 자료관리
# ============================================================

elif menu == "자료관리":
    st.subheader("📂 자료관리")
    st.write(
        "가구부문 통계조사 관련 PDF, TXT, 사진 자료를 "
        "등록할 수 있습니다."
    )
    st.divider()

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------
    st.markdown("### 📄 PDF 자료")
    pdf_files = st.file_uploader(
        "PDF 파일을 선택하세요.",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_upload"
    )

    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------
    st.markdown("### 📝 TXT 자료")
    txt_files = st.file_uploader(
        "TXT 파일을 선택하세요.",
        type=["txt"],
        accept_multiple_files=True,
        key="txt_upload"
    )

    # --------------------------------------------------------
    # 이미지
    # --------------------------------------------------------
    st.markdown("### 📷 사진 자료")
    st.write(
        "JPG, JPEG, PNG, WEBP, HEIC, HEIF 등 이미지 파일을 "
        "선택할 수 있습니다."
    )
    image_files = st.file_uploader(
        "사진 파일을 선택하세요.",
        type=["jpg", "jpeg", "png", "webp", "heic", "heif", "bmp"],
        accept_multiple_files=True,
        key="image_upload"
    )

    # --------------------------------------------------------
    # 업로드 현황
    # --------------------------------------------------------
    st.divider()
    st.subheader("📊 업로드 현황")

    pdf_count = len(pdf_files) if pdf_files else 0
    txt_count = len(txt_files) if txt_files else 0
    image_count = len(image_files) if image_files else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 PDF", pdf_count)
    with col2:
        st.metric("📝 TXT", txt_count)
    with col3:
        st.metric("📷 사진", image_count)

    # --------------------------------------------------------
    # PDF 처리
    # --------------------------------------------------------
    if pdf_files:
        st.divider()
        st.subheader("📄 PDF 내용 확인")

        for file in pdf_files:
            try:
                file.seek(0)
                pdf_bytes = file.read()
                reader = PdfReader(io.BytesIO(pdf_bytes))
                full_text = ""

                for page_number, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if text:
                        full_text += (
                            "\n\n"
                            f"===== 페이지 {page_number} ====="
                            "\n\n"
                        )
                        full_text += text

                with st.expander(f"📄 {file.name}"):
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
                        st.warning("PDF에서 텍스트를 찾지 못했습니다.")
                        st.info("스캔 PDF일 가능성이 있습니다.")
            except Exception as e:
                st.error(f"{file.name} 처리 중 오류가 발생했습니다.")
                st.code(repr(e))

    # --------------------------------------------------------
    # TXT 처리
    # --------------------------------------------------------
    if txt_files:
        st.divider()
        st.subheader("📝 TXT 내용 확인")

        for file in txt_files:
            try:
                file.seek(0)
                content_bytes = file.read()
                try:
                    content = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    content = content_bytes.decode("cp949")

                with st.expander(f"📝 {file.name}"):
                    st.text_area(
                        "TXT 내용",
                        content,
                        height=400,
                        key=f"txt_text_{file.name}"
                    )
                    st.success("TXT 파일을 정상적으로 읽었습니다.")
            except Exception as e:
                st.error(f"{file.name} 파일을 읽을 수 없습니다.")
                st.code(repr(e))

    # --------------------------------------------------------
    # 이미지 처리 + OCR
    # --------------------------------------------------------
    if image_files:
        st.divider()
        st.subheader("📷 업로드된 사진")

        for index, file in enumerate(image_files):
            with st.expander(f"📷 {file.name}", expanded=True):
                # 파일 정보
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.write("**파일명**")
                    st.code(file.name)
                with col_b:
                    st.write("**MIME 타입**")
                    st.code(str(file.type))
                with col_c:
                    st.write("**크기**")
                    st.code(f"{file.size / 1024 / 1024:.2f} MB")

                # 이미지 열기
                try:
                    file.seek(0)
                    image_bytes = file.read()

                    image = st.session_state.get(f"image_{index}_{file.name}")
                    if image is None:
                        from PIL import Image
                        image = Image.open(io.BytesIO(image_bytes))
                        image.load()
                        st.session_state[f"image_{index}_{file.name}"] = image

                    st.write("**실제 이미지 포맷**")
                    st.code(str(image.format))
                    st.image(image, caption=file.name, use_container_width=True)
                    st.success("사진을 정상적으로 읽었습니다.")

                    # OCR 실행
                    st.markdown("### 🔎 OCR (한국어 특화)")

                    if not OCR_SUPPORT:
                        st.error("RapidOCR을 사용할 수 없습니다.")
                        st.code(OCR_ERROR)
                    else:
                        if st.button(
                            "🔎 이 사진 OCR 실행",
                            key=f"ocr_button_{index}_{file.name}",
                            use_container_width=True
                        ):
                            with st.spinner("사진의 글자를 인식하고 있습니다..."):
                                result, error = run_ocr(image)

                            if error:
                                st.error("OCR 실행 중 오류가 발생했습니다.")
                                st.code(error)
                            else:
                                texts = extract_ocr_text(result)
                                if texts:
                                    st.success(
                                        f"{len(texts)}개의 "
                                        "텍스트 영역을 인식했습니다."
                                    )
                                    ocr_text = "\n".join(texts)
                                    st.text_area(
                                        "📝 OCR 인식 결과",
                                        ocr_text,
                                        height=300,
                                        key=f"ocr_result_{index}_{file.name}"
                                    )
                                else:
                                    st.warning(
                                        "OCR은 실행됐지만 "
                                        "인식된 글자가 없습니다."
                                    )

                except Exception as e:
                    st.error("이미지를 처리할 수 없습니다.")
                    st.code(repr(e))
