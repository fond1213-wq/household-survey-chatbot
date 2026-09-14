import io
import os
import sys
import uuid
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
# Supabase 지원 확인
# ============================================================

SUPABASE_SUPPORT = False
SUPABASE_ERROR = ""

try:
    from supabase import create_client
    SUPABASE_SUPPORT = True
except Exception as e:
    SUPABASE_SUPPORT = False
    SUPABASE_ERROR = repr(e)


# ============================================================
# Streamlit 설정
# ============================================================

st.set_page_config(
    page_title="가구부문통계조사 챗봇",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# 🔑 API 키 로드
# ============================================================

def _load_secret(key):
    value = None
    try:
        value = st.secrets[key]
    except Exception:
        pass
    if not value:
        value = os.getenv(key)
    return value.strip() if value else None


GEMINI_API_KEY = _load_secret("GEMINI_API_KEY")
SUPABASE_URL = _load_secret("SUPABASE_URL")
SUPABASE_KEY = _load_secret("SUPABASE_KEY")
SUPABASE_BUCKET = _load_secret("SUPABASE_BUCKET") or "chatbot-uploads"

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY가 설정되지 않았습니다.")
    st.stop()

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error(
        "SUPABASE_URL 또는 SUPABASE_KEY가 설정되지 않았습니다. "
        "자료 영구 저장을 위해 필요합니다."
    )
    st.stop()


# ============================================================
# ⚙️ 사용할 Gemini 모델
# ============================================================

GEMINI_MODEL = "gemini-3.5-flash-lite"


# ============================================================
# 🗂️ Supabase 클라이언트
# ============================================================

@st.cache_resource
def get_supabase():
    if not SUPABASE_SUPPORT:
        raise RuntimeError(SUPABASE_ERROR)
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ============================================================
# ☁️ Supabase Storage 유틸
# ============================================================

CATEGORY_PDF = "pdfs"
CATEGORY_TXT = "txts"
CATEGORY_IMAGE = "images"


def _make_safe_key(file_name: str) -> str:
    ext = ""
    if "." in file_name:
        ext = "." + file_name.rsplit(".", 1)[-1].lower()
    return f"{uuid.uuid4().hex}{ext}"


def upload_to_storage(file_bytes: bytes, file_name: str, category: str):
    supabase = get_supabase()
    safe_key = _make_safe_key(file_name)
    path = f"{category}/{safe_key}"

    supabase.storage.from_(SUPABASE_BUCKET).upload(
        path=path,
        file=file_bytes,
        file_options={
            "upsert": "true",
            "content-type": "application/octet-stream",
            "metadata": {"original_name": file_name},
        },
    )
    return safe_key, file_name


def list_storage(category: str):
    supabase = get_supabase()
    try:
        items = supabase.storage.from_(SUPABASE_BUCKET).list(category)
    except Exception:
        return []

    result = []
    for item in items or []:
        if isinstance(item, dict):
            key = item.get("name")
            metadata = item.get("metadata") or {}
        else:
            key = getattr(item, "name", None)
            metadata = getattr(item, "metadata", None) or {}

        if not key:
            continue

        if isinstance(metadata, dict):
            original_name = metadata.get("original_name", key)
        else:
            original_name = key

        result.append({"key": key, "original_name": original_name})

    return result


def download_from_storage(category: str, safe_key: str) -> bytes:
    supabase = get_supabase()
    return supabase.storage.from_(SUPABASE_BUCKET).download(
        f"{category}/{safe_key}"
    )


def delete_from_storage(category: str, safe_key: str):
    supabase = get_supabase()
    supabase.storage.from_(SUPABASE_BUCKET).remove(
        [f"{category}/{safe_key}"]
    )


def load_all_files_from_storage():
    for category, session_key in [
        (CATEGORY_PDF, "saved_pdfs"),
        (CATEGORY_TXT, "saved_txts"),
        (CATEGORY_IMAGE, "saved_images"),
    ]:
        for info in list_storage(category):
            try:
                data = download_from_storage(category, info["key"])
            except Exception:
                continue

            st.session_state[session_key][info["original_name"]] = data
            st.session_state.storage_keys[
                f"{category}/{info['original_name']}"
            ] = info["key"]


# ============================================================
# 🗂️ 세션 상태 초기화
# ============================================================

if "saved_pdfs" not in st.session_state:
    st.session_state.saved_pdfs = {}

if "saved_txts" not in st.session_state:
    st.session_state.saved_txts = {}

if "saved_images" not in st.session_state:
    st.session_state.saved_images = {}

if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = {}

if "pdf_texts" not in st.session_state:
    st.session_state.pdf_texts = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "storage_keys" not in st.session_state:
    st.session_state.storage_keys = {}

if "storage_loaded" not in st.session_state:
    with st.spinner("☁️ 저장된 자료를 불러오는 중..."):
        try:
            load_all_files_from_storage()
        except Exception as e:
            st.warning(f"저장된 자료 로드 실패: {e}")
    st.session_state.storage_loaded = True


# ============================================================
# 📄 PDF 텍스트 추출 (캐시)
# ============================================================

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """PDF에서 전체 텍스트를 추출합니다."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = ""
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                full_text += f"\n\n===== 페이지 {page_number} =====\n\n"
                full_text += text
        return full_text
    except Exception as e:
        return f"[PDF 파싱 오류: {e}]"


def get_pdf_text_cached(name: str, pdf_bytes: bytes) -> str:
    """세션 캐시를 활용하여 PDF 텍스트를 반환합니다."""
    if name not in st.session_state.pdf_texts:
        st.session_state.pdf_texts[name] = extract_pdf_text(pdf_bytes)
    return st.session_state.pdf_texts[name]


# ============================================================
# 📚 문서 컨텍스트 빌더 (RAG 핵심)
# ============================================================

def build_document_context(max_chars: int = 400_000) -> str:
    """
    업로드된 모든 문서(PDF, TXT, 이미지 OCR)의 텍스트를
    하나의 컨텍스트 문자열로 결합합니다.
    """
    parts = []
    total = 0

    # --- PDF ---
    for name, pdf_bytes in st.session_state.saved_pdfs.items():
        text = get_pdf_text_cached(name, pdf_bytes)
        if not text.strip():
            continue
        block = f"\n\n========== [PDF 문서: {name}] ==========\n{text}"
        if total + len(block) > max_chars:
            block = block[: max_chars - total]
        parts.append(block)
        total += len(block)
        if total >= max_chars:
            break

    # --- TXT ---
    if total < max_chars:
        for name, content_bytes in st.session_state.saved_txts.items():
            try:
                try:
                    content = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    content = content_bytes.decode("cp949")
            except Exception:
                continue
            block = f"\n\n========== [TXT 문서: {name}] ==========\n{content}"
            if total + len(block) > max_chars:
                block = block[: max_chars - total]
            parts.append(block)
            total += len(block)
            if total >= max_chars:
                break

    # --- 이미지 OCR 결과 ---
    if total < max_chars:
        for key, text in st.session_state.ocr_results.items():
            if not text.strip():
                continue
            block = f"\n\n========== [이미지 OCR: {key}] ==========\n{text}"
            if total + len(block) > max_chars:
                block = block[: max_chars - total]
            parts.append(block)
            total += len(block)
            if total >= max_chars:
                break

    return "".join(parts)


# ============================================================
# 🤖 Gemini 클라이언트
# ============================================================

@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)


def call_gemini(
    question: str,
    model: str = GEMINI_MODEL,
    context: str = None,
    system_prompt: str = None,
) -> str:
    """
    Gemini API를 호출합니다.
    context가 주어지면 문서 기반 답변을 생성합니다.
    """
    if system_prompt is None:
        system_prompt = (
            "당신은 가구부문 통계조사 업무를 돕는 AI 어시스턴트입니다.\n"
            "**반드시 아래 제공된 문서 내용을 기반으로만 답변하세요.**\n"
            "등록된 문서에 없는 내용은 추측하지 말고 "
            "'등록된 지침서에서 해당 내용을 찾을 수 없습니다'라고 답변하세요.\n"
            "여러 문서가 제공된 경우, 질문과 가장 관련 있는 문서를 우선 참고하세요.\n"
            "한국어로 정확하고 친절하게 답변해주세요."
        )

    if context and context.strip():
        full_prompt = (
            f"[참고 문서]\n{context}\n\n"
            f"---\n\n"
            f"[질문]\n{question}"
        )
    else:
        full_prompt = question

    client = get_gemini_client()
    response = client.models.generate_content(
        model=model,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
            max_output_tokens=2000,
        ),
    )
    return response.text


# ============================================================
# 🔬 진단
# ============================================================

def test_gemini_model() -> dict:
    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents="ping",
        )
        return {
            "ok": True,
            "model": GEMINI_MODEL,
            "message": response.text[:100],
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "model": GEMINI_MODEL,
            "message": None,
            "error": repr(e),
        }


def test_supabase() -> dict:
    try:
        supabase = get_supabase()
        buckets = supabase.storage.list_buckets()
        names = []
        for b in buckets or []:
            names.append(getattr(b, "name", str(b)))
        return {"ok": True, "buckets": names, "error": None}
    except Exception as e:
        return {"ok": False, "buckets": [], "error": repr(e)}


# ============================================================
# OCR 엔진
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


def run_ocr(pil_image):
    try:
        import numpy as np
        image_array = np.array(pil_image.convert("RGB"))
        engine = get_ocr_engine()
        result = engine(image_array)
        return result, None
    except Exception as e:
        return None, repr(e)


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
# 제목 + 🔴 답변 범위 안내 배너
# ============================================================

st.title("📚 가구부문통계조사 챗봇")
st.caption(
    "가구부문 통계조사 업무자료를 기반으로 "
    "질문에 답변하는 AI 챗봇"
)

st.markdown(
    """
    <div style="
        background-color: #FDECEC;
        border-left: 6px solid #D32F2F;
        border-radius: 8px;
        padding: 18px 22px;
        margin: 18px 0;
        box-shadow: 0 2px 8px rgba(211, 47, 47, 0.15);
    ">
        <div style="
            font-size: 1.1rem;
            font-weight: 800;
            color: #B71C1C;
            margin-bottom: 10px;
        ">
            ⚠️ 답변 범위 안내
        </div>
        <div style="
            font-size: 0.98rem;
            color: #7F1D1D;
            line-height: 1.7;
            font-weight: 500;
        ">
            본 챗봇은 <b>아래 등록된 지침서 기반으로만 답변</b>합니다.<br>
            · <b>2026년 경제활동인구조사 지침서</b> (외부용)<br>
            · <b>2024년 가계동향조사 지침서</b>
        </div>
        <div style="
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px dashed #F5A5A5;
            font-size: 0.88rem;
            color: #991B1B;
        ">
            ※ 등록되지 않은 내용은 답변하지 않거나 안내가 제한될 수 있습니다.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# 사이드바
# ============================================================

with st.sidebar:
    st.header("📂 메뉴")
    menu = st.radio("이동", ["질문하기", "자료관리"])
    st.divider()

    st.write("**HEIC 지원**")
    if HEIC_SUPPORT:
        st.success("✅ 지원")
    else:
        st.warning("❌ 지원 안 됨")

    st.write("**OCR 지원**")
    if OCR_SUPPORT:
        st.success("✅ RapidOCR v3")
    else:
        st.error("❌ 지원 안 됨")

    st.write("**영구 저장 (Supabase)**")
    if SUPABASE_SUPPORT:
        st.success("✅ 연결됨")
    else:
        st.error("❌ 미지원")

    st.write("**AI 모델**")
    st.code(GEMINI_MODEL)

    # 문서 현황 요약
    st.write("**등록된 문서**")
    st.write(f"- PDF: {len(st.session_state.saved_pdfs)}개")
    st.write(f"- TXT: {len(st.session_state.saved_txts)}개")
    st.write(f"- 사진: {len(st.session_state.saved_images)}개")
    st.write(f"- OCR 결과: {len(st.session_state.ocr_results)}개")

    st.divider()

    with st.expander("🔧 시스템 진단"):
        st.write("Python 버전")
        st.code(sys.version)
        st.write("RapidOCR 버전")
        st.code(RAPIDOCR_VERSION)
        st.write("ONNX Runtime 버전")
        st.code(ONNXRUNTIME_VERSION)
        st.write("Gemini 키 앞 8자리")
        st.code(GEMINI_API_KEY[:8] if GEMINI_API_KEY else "없음")
        st.write("Supabase URL")
        st.code(SUPABASE_URL or "없음")
        st.write("Supabase 버킷")
        st.code(SUPABASE_BUCKET)

    st.divider()

    if st.button("🔑 Gemini 테스트", use_container_width=True):
        with st.spinner("테스트 중..."):
            r = test_gemini_model()
        if r["ok"]:
            st.success(f"✅ {r['model']}")
            st.caption(f"응답: {r['message']}")
        else:
            st.error("❌ 실패")
            st.code(r["error"])

    if st.button("☁️ Supabase 테스트", use_container_width=True):
        with st.spinner("테스트 중..."):
            r = test_supabase()
        if r["ok"]:
            st.success("✅ 연결 성공")
            st.write("버킷 목록:")
            for name in r["buckets"]:
                st.code(name)
        else:
            st.error("❌ 실패")
            st.code(r["error"])

    if st.button("🔄 저장소 새로고침", use_container_width=True):
        st.session_state.saved_pdfs = {}
        st.session_state.saved_txts = {}
        st.session_state.saved_images = {}
        st.session_state.ocr_results = {}
        st.session_state.pdf_texts = {}
        st.session_state.storage_keys = {}
        st.session_state.storage_loaded = False
        st.rerun()


# ============================================================
# 질문하기
# ============================================================

if menu == "질문하기":
    st.subheader("💬 질문하기")
    st.info(f"🤖 사용 모델: **{GEMINI_MODEL}**")

    # 현재 등록된 문서 정보 표시
    doc_count = (
        len(st.session_state.saved_pdfs)
        + len(st.session_state.saved_txts)
        + len(st.session_state.ocr_results)
    )
    if doc_count > 0:
        st.success(f"📚 현재 {doc_count}개의 문서가 등록되어 있습니다. 문서 기반으로 답변합니다.")
    else:
        st.warning("⚠️ 등록된 문서가 없습니다. '자료관리' 탭에서 자료를 먼저 등록해주세요.")

    question = st.text_area(
        "궁금한 내용을 입력하세요.",
        placeholder="예: 취업자는 어떤 기준으로 판단하나요?",
        height=120
    )

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        ask_clicked = st.button("🔍 질문하기", use_container_width=True)
    with col_btn2:
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if ask_clicked:
        if question.strip():
            with st.spinner(f"{GEMINI_MODEL} 모델로 답변 생성 중..."):
                try:
                    # 모든 문서를 컨텍스트로 결합
                    context = build_document_context()
                    answer = call_gemini(
                        question,
                        model=GEMINI_MODEL,
                        context=context if context.strip() else None,
                    )
                    st.session_state.chat_history.append(
                        {"role": "user", "content": question}
                    )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.success("답변")
                    st.write(answer)

                    with st.expander("📄 참고한 문서 정보"):
                        st.write(f"컨텍스트 길이: {len(context):,}자")
                        st.write(f"PDF: {len(st.session_state.saved_pdfs)}개")
                        st.write(f"TXT: {len(st.session_state.saved_txts)}개")
                        st.write(f"OCR: {len(st.session_state.ocr_results)}개")
                except Exception as e:
                    st.error("Gemini 호출 중 오류가 발생했습니다.")
                    st.code(repr(e))
        else:
            st.warning("질문을 입력해주세요.")

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
        "등록할 수 있습니다. 업로드된 자료는 **Supabase에 "
        "영구 저장**되어 앱을 재시작해도 유지됩니다."
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
            data = f.read()
            try:
                safe_key, original = upload_to_storage(
                    data, f.name, CATEGORY_PDF
                )
                st.session_state.saved_pdfs[original] = data
                st.session_state.storage_keys[
                    f"{CATEGORY_PDF}/{original}"
                ] = safe_key
                # PDF 텍스트 캐시 생성
                if original not in st.session_state.pdf_texts:
                    st.session_state.pdf_texts[original] = extract_pdf_text(data)
            except Exception as e:
                st.error(f"{f.name} 업로드 실패: {e}")

    # PDF 목록 + 개별 삭제
    if st.session_state.saved_pdfs:
        st.caption(f"📦 저장된 PDF: {len(st.session_state.saved_pdfs)}개")

        for name in list(st.session_state.saved_pdfs.keys()):
            col_name, col_del = st.columns([5, 1])
            with col_name:
                text_len = len(st.session_state.pdf_texts.get(name, ""))
                st.write(f"📄 **{name}** ({text_len:,}자 추출됨)")
            with col_del:
                if st.button("🗑️", key=f"del_pdf_{name}", help="이 PDF 삭제"):
                    try:
                        safe_key = st.session_state.storage_keys.get(
                            f"{CATEGORY_PDF}/{name}"
                        )
                        if safe_key:
                            delete_from_storage(CATEGORY_PDF, safe_key)
                    except Exception:
                        pass
                    st.session_state.saved_pdfs.pop(name, None)
                    st.session_state.pdf_texts.pop(name, None)
                    st.session_state.storage_keys.pop(
                        f"{CATEGORY_PDF}/{name}", None
                    )
                    st.rerun()

        if st.button("🗑️ PDF 전체 삭제", key="clear_pdfs"):
            for name in list(st.session_state.saved_pdfs.keys()):
                try:
                    safe_key = st.session_state.storage_keys.get(
                        f"{CATEGORY_PDF}/{name}"
                    )
                    if safe_key:
                        delete_from_storage(CATEGORY_PDF, safe_key)
                except Exception:
                    pass
            st.session_state.saved_pdfs = {}
            st.session_state.pdf_texts = {}
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
            data = f.read()
            try:
                safe_key, original = upload_to_storage(
                    data, f.name, CATEGORY_TXT
                )
                st.session_state.saved_txts[original] = data
                st.session_state.storage_keys[
                    f"{CATEGORY_TXT}/{original}"
                ] = safe_key
            except Exception as e:
                st.error(f"{f.name} 업로드 실패: {e}")

    if st.session_state.saved_txts:
        st.caption(f"📦 저장된 TXT: {len(st.session_state.saved_txts)}개")

        for name in list(st.session_state.saved_txts.keys()):
            col_name, col_del = st.columns([5, 1])
            with col_name:
                st.write(f"📝 **{name}**")
            with col_del:
                if st.button("🗑️", key=f"del_txt_{name}", help="이 TXT 삭제"):
                    try:
                        safe_key = st.session_state.storage_keys.get(
                            f"{CATEGORY_TXT}/{name}"
                        )
                        if safe_key:
                            delete_from_storage(CATEGORY_TXT, safe_key)
                    except Exception:
                        pass
                    st.session_state.saved_txts.pop(name, None)
                    st.session_state.storage_keys.pop(
                        f"{CATEGORY_TXT}/{name}", None
                    )
                    st.rerun()

        if st.button("🗑️ TXT 전체 삭제", key="clear_txts"):
            for name in list(st.session_state.saved_txts.keys()):
                try:
                    safe_key = st.session_state.storage_keys.get(
                        f"{CATEGORY_TXT}/{name}"
                    )
                    if safe_key:
                        delete_from_storage(CATEGORY_TXT, safe_key)
                except Exception:
                    pass
            st.session_state.saved_txts = {}
            st.rerun()

    # --------------------------------------------------------
    # 이미지 업로더
    # --------------------------------------------------------
    st.markdown("### 📷 사진 자료")
    image_files = st.file_uploader(
        "사진 파일을 선택하세요.",
        type=["jpg", "jpeg", "png", "webp", "heic", "heif", "bmp"],
        accept_multiple_files=True,
        key="image_upload"
    )

    if image_files:
        for f in image_files:
            f.seek(0)
            data = f.read()
            try:
                safe_key, original = upload_to_storage(
                    data, f.name, CATEGORY_IMAGE
                )
                st.session_state.saved_images[original] = data
                st.session_state.storage_keys[
                    f"{CATEGORY_IMAGE}/{original}"
                ] = safe_key
            except Exception as e:
                st.error(f"{f.name} 업로드 실패: {e}")

    if st.session_state.saved_images:
        st.caption(f"📦 저장된 사진: {len(st.session_state.saved_images)}개")

        for name in list(st.session_state.saved_images.keys()):
            col_name, col_del = st.columns([5, 1])
            with col_name:
                st.write(f"📷 **{name}**")
            with col_del:
                if st.button("🗑️", key=f"del_img_{name}", help="이 사진 삭제"):
                    try:
                        safe_key = st.session_state.storage_keys.get(
                            f"{CATEGORY_IMAGE}/{name}"
                        )
                        if safe_key:
                            delete_from_storage(CATEGORY_IMAGE, safe_key)
                    except Exception:
                        pass
                    st.session_state.saved_images.pop(name, None)
                    st.session_state.storage_keys.pop(
                        f"{CATEGORY_IMAGE}/{name}", None
                    )
                    # 관련 OCR 결과도 삭제
                    for k in list(st.session_state.ocr_results.keys()):
                        if name in k:
                            st.session_state.ocr_results.pop(k, None)
                    st.rerun()

        if st.button("🗑️ 사진 전체 삭제", key="clear_images"):
            for name in list(st.session_state.saved_images.keys()):
                try:
                    safe_key = st.session_state.storage_keys.get(
                        f"{CATEGORY_IMAGE}/{name}"
                    )
                    if safe_key:
                        delete_from_storage(CATEGORY_IMAGE, safe_key)
                except Exception:
                    pass
            st.session_state.saved_images = {}
            st.session_state.ocr_results = {}
            st.rerun()

    # --------------------------------------------------------
    # 업로드 현황
    # --------------------------------------------------------
    st.divider()
    st.subheader("📊 업로드 현황")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 PDF", len(st.session_state.saved_pdfs))
    with col2:
        st.metric("📝 TXT", len(st.session_state.saved_txts))
    with col3:
        st.metric("📷 사진", len(st.session_state.saved_images))
    with col4:
        st.metric("🔎 OCR", len(st.session_state.ocr_results))

    # --------------------------------------------------------
    # PDF 내용 확인
    # --------------------------------------------------------
    if st.session_state.saved_pdfs:
        st.divider()
        st.subheader("📄 PDF 내용 확인")

        for name, pdf_bytes in st.session_state.saved_pdfs.items():
            try:
                full_text = get_pdf_text_cached(name, pdf_bytes)
                reader = PdfReader(io.BytesIO(pdf_bytes))

                with st.expander(f"📄 {name}"):
                    if full_text.strip():
                        st.text_area(
                            "추출된 텍스트",
                            full_text,
                            height=500,
                            key=f"pdf_text_{name}"
                        )
                        st.success(
                            f"{len(reader.pages)}페이지, "
                            f"{len(full_text):,}자 추출됨"
                        )
                    else:
                        st.warning("PDF에서 텍스트를 찾지 못했습니다.")
            except Exception as e:
                st.error(f"{name} 처리 중 오류")
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
            with st.expander(f"📷 {name}", expanded=False):
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
                            with st.spinner("사진의 글자를 인식하고 있습니다..."):
                                result, error = run_ocr(image)

                            if error:
                                st.error("OCR 실행 중 오류")
                                st.code(error)
                            else:
                                texts = extract_ocr_text(result)
                                if texts:
                                    st.session_state.ocr_results[ocr_key] = "\n".join(texts)
                                    st.success(
                                        f"{len(texts)}개의 텍스트 영역을 인식했습니다."
                                    )
                                else:
                                    st.warning("인식된 글자가 없습니다.")

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
