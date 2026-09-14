import io
import streamlit as st
from pypdf import PdfReader

# ------------------------------------------------------------
# HEIC/HEIF 지원 (설치돼 있으면 자동 등록, 없으면 무시)
#   pip install pillow-heif
# ------------------------------------------------------------
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except Exception:
    HEIC_SUPPORT = False


# ============================================================
# 기본 설정
# ============================================================

st.set_page_config(
    page_title="가구부문통계조사 챗봇",
    page_icon="📚",
    layout="wide"
)


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
        [
            "질문하기",
            "자료관리"
        ]
    )

    st.divider()

    st.caption(
        f"HEIC 지원: {'✅' if HEIC_SUPPORT else '❌ (pip install pillow-heif)'}"
    )


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

    if st.button(
        "🔍 질문하기",
        use_container_width=True
    ):

        if question.strip():

            st.info(
                "현재는 문서 처리 단계입니다. "
                "추후 AI가 등록된 자료를 검색하여 답변하도록 연결합니다."
            )

            st.write("입력하신 질문:")
            st.write(question)

        else:

            st.warning(
                "질문을 입력해주세요."
            )


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


    # ========================================================
    # PDF
    # ========================================================

    st.markdown("### 📄 PDF 자료")

    pdf_files = st.file_uploader(
        "PDF 파일을 선택하세요.",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_upload",
    )


    # ========================================================
    # TXT
    # ========================================================

    st.markdown("### 📝 TXT 자료")

    txt_files = st.file_uploader(
        "TXT 파일을 선택하세요.",
        type=["txt"],
        accept_multiple_files=True,
        key="txt_upload",
    )


    # ========================================================
    # 사진
    # ========================================================

    st.markdown("### 📷 사진 자료")

    st.write(
        "JPG, JPEG, PNG, WEBP, HEIC, HEIF 등 이미지 파일을 "
        "선택할 수 있습니다."
    )

    image_files = st.file_uploader(
        "사진 파일을 선택하세요.",
        type=["jpg", "jpeg", "png", "webp", "heic", "heif", "bmp"],
        accept_multiple_files=True,
        key="image_upload",
    )


    # ========================================================
    # 업로드 현황
    # ========================================================

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


    # ========================================================
    # PDF 내용 확인
    # ========================================================

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
                st.code(str(e))


    # ========================================================
    # TXT 내용 확인
    # ========================================================

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

                    st.success(
                        "TXT 파일을 정상적으로 읽었습니다."
                    )

            except Exception as e:

                st.error(
                    f"{file.name} 파일을 읽을 수 없습니다."
                )
                st.code(str(e))


    # ========================================================
    # 사진 확인
    # ========================================================

    if image_files:

        st.divider()

        st.subheader("📷 업로드된 사진")

        for index, file in enumerate(image_files):

            with st.expander(
                f"📷 {file.name}",
                expanded=True
            ):

                # 파일 기본 정보
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

                # 이미지 표시 시도
                try:

                    file.seek(0)
                    image_bytes = file.read()
                    file.seek(0)

                    # PIL로 열어서 포맷 확인 + Streamlit에 전달
                    try:
                        from PIL import Image
                        pil_img = Image.open(io.BytesIO(image_bytes))
                        fmt = pil_img.format  # JPEG, PNG, HEIF 등

                        st.write("**실제 포맷**")
                        st.code(str(fmt))

                        st.image(
                            pil_img,
                            caption=file.name,
                            use_container_width=True
                        )

                        st.success(
                            "사진이 정상적으로 업로드되었습니다."
                        )

                    except Exception as pil_err:
                        # PIL로 못 여는 경우 (HEIC 미지원 등)
                        st.warning(
                            "이미지를 열 수 없습니다. "
                            "HEIC라면 `pip install pillow-heif` 후 다시 시도하세요."
                        )
                        st.code(str(pil_err))

                        # 원본 바이트로 한 번 더 시도
                        st.image(
                            image_bytes,
                            caption=file.name,
                            use_container_width=True
                        )

                except Exception as e:

                    st.warning(
                        "파일은 업로드되었지만 "
                        "현재 화면에서 이미지를 표시할 수 없습니다."
                    )
                    st.code(str(e))

                st.info(
                    "OCR 기능은 다음 단계에서 연결합니다."
                )
