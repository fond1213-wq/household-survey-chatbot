import io
import os
import sys
import importlib.metadata

import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types

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
# RapidOCR 지원 확인
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
# Streamlit 설정
# ============================================================

st.set_page_config(
    page_title="가구부문통계조사 챗봇",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# 🔑 API 키 로드 (Secrets 우선, 환경변수 폴백)
# ============================================================

GEMINI_API_KEY = None

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY가 설정되지 않았습니다.")
    with st.expander("🔧 진단"):
        try:
            st.write("Secrets 키 목록:", list(st.secrets.keys()))
        except Exception as e:
            st.write("st.secrets 오류:", repr(e))
    st.stop()

GEMINI_API_KEY = GEMINI_API_KEY.strip()


# ============================================================
# 🗂️ 세션 상태 초기화 (업로드 파일 보존)
# ============================================================

if "saved_pdfs" not in st.session_state:
    st.session_state.saved_pdfs = {}

if "saved_txts" not in st.session_state:
    st.session_state.saved_txts = {}

if "saved_images" not in st.session_state:
    st.session_state.saved_images = {}

if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# 🤖 Gemini 클라이언트 (google-genai SDK)
# ============================================================

@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)


def call_gemini(
    question: str,
    model: str = "gemini-2.5-flash-lite",
    system_prompt: str = None,
) -> str:
    """
    Gemini API를 호출하여 답변을 반환합니다.
    """
    if system_prompt is None:
        system_prompt = (
            "당신은 가구부문 통계조사 업무를 돕는 "
            "친절한 AI 어시스턴트입니다. "
            "한국어로 정확하게 답변해주세요."
        )

    client = get_gemini_client()

    response = client.models.generate_content(
        model=model,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
            max_output_tokens=2000,
        ),
    )

    return response.text


def test_gemini_auth() -> dict:
    """Gemini API 인증 상태를 테스트합니다."""
    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents="ping",
        )
        return {
            "ok": True,
            "message": response.text[:100],
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "message": None,
            "error": repr(e),
        }


# ============================================================
# OCR 엔진 (한국어 특화)
# ============================================================

@st.cache_resource
def get_ocr_engine():
    if not OCR_SUPPORT:
        raise RuntimeError(OCR_ERROR)

    try:
        engine = RapidOCR(
            params={
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Det.lang_type": LangDet.MULTI,
                "Det.model_type": ModelType.MOBILE,
                "Det.ocr_version": OCRVersion.PPOCRV5,

                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.lang_type": LangRec.KOREAN,
                "Rec.model_type": ModelType.MOBILE,
                "Rec.ocr_version": OCRVersion.PPOCRV5,

                "Cls.engine_type": EngineType.ONNXRUNTIME,
                "Cls.lang_type": LangDet.CH,
                "Cls.model_type": ModelType.MOBILE,
                "Cls.ocr_version": OCRVersion.PPOCRV4,
            }
        )
        return engine

    except Exception as e:
        try:
            st.warning(f"한국어 모델 로드 실패. 기본 모델로 대체합니다. (원인: {e})")
            engine = RapidOCR(
                params={
                    "Det.engine_type": EngineType.ONNXRUNTIME,
                    "Det.lang_type": LangDet.CH,
                    "Det.model_type": ModelType.MOBILE,
                    "Det.ocr_version": OCRVersion.PPOCRV5,
                    "Rec.engine_type": EngineType.ONNXRUNTIME,
                    "Rec.lang_type": LangRec.CH,
                    "Rec.model_type": ModelType.MOBILE,
                    "Rec.ocr_version": OCRVersion.PPOCRV5,
                    "Cls.engine_type": EngineType.ONNXRUNTIME,
                    "Cls.lang_type": LangDet.CH,
                    "Cls.model_type": ModelType.MOBILE,
                    "Cls.ocr_version": OCRVersion.PPOCRV4,
                }
            )
            return engine
        except Exception as e2:
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

        st.write("Gemini API 키 앞 8자리")
        st.code(GEMINI_API_KEY[:8] if GEMINI_API_KEY else "없음")

        st.write("Gemini API 키 길이")
        st.code(str(len(GEMINI_API_KEY)) if GEMINI_API_KEY else "0")

        if OCR_SUPPORT:
            st.success("RapidOCR import 성공")
        else:
            st.error("RapidOCR import 실패")
            st.code(OCR_ERROR)

    st.divider()

    if st.button("🔑 Gemini 인증 테스트", use_container_width=True):
        with st.spinner("인증 확인 중..."):
            result = test_gemini_auth()

        if result["ok"]:
            st.success("✅ Gemini API 인증 성공")
            st.caption(f"응답: {result['message']}")
        else:
            st.error("❌ Gemini 인증 실패")
            st.code(result["error"])


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

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        ask_clicked = st.button(
            "🔍 질문하기", use_container_width=True
        )
    with col_btn2:
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if ask_clicked:
        if question.strip():
            with st.spinner("Gemini가 답변을 생성하고 있습니다..."):
                try:
                    answer = call_gemini(question)

                    st.session_state.chat_history.append(
                        {"role": "user", "content": question}
                    )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )

                    st.success("답변")
                    st.write(answer)

                except Exception as e:
                    st.error("Gemini 호출 중 오류가 발생했습니다.")
                    st.code(repr(e))

                    with st.expander("🔧 인증 오류 진단"):
                        st.write(
                            "키 앞 8자리:",
                            GEMINI_API_KEY[:8] if GEMINI_API_KEY else "없음"
                        )
                        st.write(
                            "키 길이:",
                            len(GEMINI_API_KEY) if GEMINI_API_KEY else 0
                        )
                        st.write("모델명: gemini-2.5-flash-lite")
                        st.info(
                            "401/403 오류가 계속되면:\n"
                            "1. aistudio.google.com/app/apikey 에서 키가 유효한지 확인\n"
                            "2. 키를 재발급하고 Streamlit Secrets 업데이트\n"
                            "3. ⋮ → Reboot app 클릭\n"
                            "4. Secrets 키 이름이 정확히 GEMINI_API_KEY 인지 확인\n"
                            "5. 사이드바의 '🔑 Gemini 인증 테스트' 버튼으로 재확인"
                        )
        else:
            st.warning("질문을 입력해주세요.")

    # 대화 기록 표시
    if st.session_state.chat_history:
        st.divider()
        st.subheader("💬 대화 기록")
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**🙋 질문:** {msg['content']}")
            else:
                st.markdown(f"**🤖 답변:** {msg['content']}")
            st.divider()


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
    # PDF 업로더
    # --------------------------------------------------------
    st.markdown("### 📄 PDF 자료")
    pdf_files = st.file_uploader(
        "PDF 파일을 선택하세요.",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_upload"
    )

    if pdf_files:
        for f in pdf_files:
            f.seek(0)
            st.session_state.saved_pdfs[f.name] = f.read()

    if st.session_state.saved_pdfs:
        st.caption(f"📦 저장된 PDF: {len(st.session_state.saved_pdfs)}개")
        if st.button("🗑️ PDF 목록 비우기", key="clear_pdfs"):
            st.session_state.saved_pdfs = {}
            st.rerun()

    # --------------------------------------------------------
    # TXT 업로더
    # --------------------------------------------------------
    st.markdown("### 📝 TXT 자료")
    txt_files = st.file_uploader(
        "TXT 파일을 선택하세요.",
        type=["txt"],
        accept_multiple_files=True,
        key="txt_upload"
    )

    if txt_files:
        for f in txt_files:
            f.seek(0)
            st.session_state.saved_txts[f.name] = f.read()

    if st.session_state.saved_txts:
        st.caption(f"📦 저장된 TXT: {len(st.session_state.saved_txts)}개")
        if st.button("🗑️ TXT 목록 비우기", key="clear_txts"):
            st.session_state.saved_txts = {}
            st.rerun()

    # --------------------------------------------------------
    # 이미지 업로더
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

    if image_files:
        for f in image_files:
            f.seek(0)
            st.session_state.saved_images[f.name] = f.read()

    if st.session_state.saved_images:
        st.caption(f"📦 저장된 사진: {len(st.session_state.saved_images)}개")
        if st.button("🗑️ 사진 목록 비우기", key="clear_images"):
            st.session_state.saved_images = {}
            st.session_state.ocr_results = {}
            st.rerun()

    # --------------------------------------------------------
    # 업로드 현황
    # --------------------------------------------------------
    st.divider()
    st.subheader("📊 업로드 현황")

    pdf_count = len(st.session_state.saved_pdfs)
    txt_count = len(st.session_state.saved_txts)
    image_count = len(st.session_state.saved_images)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 PDF", pdf_count)
    with col2:
        st.metric("📝 TXT", txt_count)
    with col3:
        st.metric("📷 사진", image_count)

    # --------------------------------------------------------
    # PDF 내용 확인
    # --------------------------------------------------------
    if st.session_state.saved_pdfs:
        st.divider()
        st.subheader("📄 PDF 내용 확인")

        for name, pdf_bytes in st.session_state.saved_pdfs.items():
            try:
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

                with st.expander(f"📄 {name}"):
                    if full_text.strip():
                        st.text_area(
                            "추출된 텍스트",
                            full_text,
                            height=500,
                            key=f"pdf_text_{name}"
                        )
                        st.success(
                            f"{len(reader.pages)}페이지에서 "
                            "텍스트를 추출했습니다."
                        )
                    else:
                        st.warning("PDF에서 텍스트를 찾지 못했습니다.")
                        st.info("스캔 PDF일 가능성이 있습니다.")
            except Exception as e:
                st.error(f"{name} 처리 중 오류가 발생했습니다.")
                st.code(repr(e))

    # --------------------------------------------------------
    # TXT 내용 확인
    # --------------------------------------------------------
    if st.session_state.saved_txts:
        st.divider()
        st.subheader("📝 TXT 내용 확인")

        for name, content_bytes in st.session_state.saved_txts.items():
            try:
                try:
                    content = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    content = content_bytes.decode("cp949")

                with st.expander(f"📝 {name}"):
                    st.text_area(
                        "TXT 내용",
                        content,
                        height=400,
                        key=f"txt_text_{name}"
                    )
                    st.success("TXT 파일을 정상적으로 읽었습니다.")
            except Exception as e:
                st.error(f"{name} 파일을 읽을 수 없습니다.")
                st.code(repr(e))

    # --------------------------------------------------------
    # 이미지 + OCR
    # --------------------------------------------------------
    if st.session_state.saved_images:
        st.divider()
        st.subheader("📷 업로드된 사진")

        from PIL import Image

        for index, (name, image_bytes) in enumerate(
            st.session_state.saved_images.items()
        ):
            with st.expander(f"📷 {name}", expanded=True):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.write("**파일명**")
                    st.code(name)
                with col_b:
                    st.write("**크기**")
                    st.code(f"{len(image_bytes) / 1024 / 1024:.2f} MB")
                with col_c:
                    st.write("**포맷**")
                    try:
                        _img = Image.open(io.BytesIO(image_bytes))
                        st.code(str(_img.format))
                    except Exception:
                        st.code("알 수 없음")

                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    image.load()

                    st.image(image, caption=name, use_container_width=True)
                    st.success("사진을 정상적으로 읽었습니다.")

                    st.markdown("### 🔎 OCR (한국어 특화)")

                    if not OCR_SUPPORT:
                        st.error("RapidOCR을 사용할 수 없습니다.")
                        st.code(OCR_ERROR)
                    else:
                        ocr_key = f"ocr_{index}_{name}"

                        if st.button(
                            "🔎 이 사진 OCR 실행",
                            key=f"ocr_button_{index}_{name}",
                            use_container_width=True
                        ):
                            with st.spinner(
                                "사진의 글자를 인식하고 있습니다..."
                            ):
                                result, error = run_ocr(image)

                            if error:
                                st.error("OCR 실행 중 오류가 발생했습니다.")
                                st.code(error)
                            else:
                                texts = extract_ocr_text(result)
                                if texts:
                                    st.session_state.ocr_results[
                                        ocr_key
                                    ] = "\n".join(texts)
                                    st.success(
                                        f"{len(texts)}개의 "
                                        "텍스트 영역을 인식했습니다."
                                    )
                                else:
                                    st.warning(
                                        "OCR은 실행됐지만 "
                                        "인식된 글자가 없습니다."
                                    )

                        if ocr_key in st.session_state.ocr_results:
                            st.text_area(
                                "📝 OCR 인식 결과",
                                st.session_state.ocr_results[ocr_key],
                                height=300,
                                key=f"ocr_result_{index}_{name}"
                            )

                except Exception as e:
                    st.error("이미지를 처리할 수 없습니다.")
                    st.code(repr(e))
