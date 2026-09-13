import streamlit as st
import io

st.set_page_config(
    page_title="가구부문통계조사 챗봇",
    page_icon="📚",
    layout="wide"
)

st.title("📚 가구부문통계조사 챗봇")

st.caption(
    "가구부문 통계조사 업무자료를 기반으로 질문에 답변하는 AI 챗봇"
)

st.divider()

# =============================
# 사이드바
# =============================

with st.sidebar:
    st.header("📂 메뉴")

    menu = st.radio(
        "이동",
        ["질문하기", "자료관리"]
    )


# =============================
# 질문하기
# =============================

if menu == "질문하기":

    st.subheader("💬 질문하기")

    question = st.text_area(
        "궁금한 내용을 입력하세요.",
        placeholder="예: 취업자는 어떤 기준으로 판단하나요?",
        height=120
    )

    if st.button("🔍 질문하기", use_container_width=True):

        if question.strip():
            st.info("현재는 문서 처리 단계입니다.")
            st.write("입력하신 질문:")
            st.write(question)

        else:
            st.warning("질문을 입력해주세요.")


# =============================
# 자료관리
# =============================

elif menu == "자료관리":

    st.subheader("📂 자료관리")

    st.write(
        "PDF, TXT, 사진 자료를 업로드하여 "
        "가구부문통계조사 지식자료로 사용할 수 있습니다."
    )

    st.divider()

    # -------------------------
    # PDF
    # -------------------------

    st.markdown("### 📄 PDF 자료")

    pdf_files = st.file_uploader(
        "PDF 파일을 선택하세요.",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_upload"
    )

    # -------------------------
    # TXT
    # -------------------------

    st.markdown("### 📝 TXT 자료")

    txt_files = st.file_uploader(
        "TXT 파일을 선택하세요.",
        type=["txt"],
        accept_multiple_files=True,
        key="txt_upload"
    )

    # -------------------------
    # 사진
    # -------------------------

    st.markdown("### 📷 사진 자료")

    image_files = st.file_uploader(
        "사진 파일을 선택하세요.",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="image_upload"
    )

    st.divider()

    # =============================
    # 업로드 현황
    # =============================

    st.subheader("📊 업로드 현황")

    pdf_count = len(pdf_files) if pdf_files else 0
    txt_count = len(txt_files) if txt_files else 0
    image_count = len(image_files) if image_files else 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("PDF", pdf_count)

    with col2:
        st.metric("TXT", txt_count)

    with col3:
        st.metric("사진", image_count)

    # =============================
    # TXT 내용 확인
    # =============================

    if txt_files:

        st.divider()
        st.subheader("📝 TXT 내용 확인")

        for file in txt_files:

            try:
                content = file.read().decode("utf-8")

                with st.expander(file.name):
                    st.text(content)

            except Exception:
                st.error(
                    f"{file.name} 파일을 읽을 수 없습니다."
                )

    # =============================
    # 사진 확인
    # =============================

    if image_files:

        st.divider()
        st.subheader("📷 사진 확인")

        for file in image_files:

            with st.expander(file.name):
                st.image(
                    file,
                    caption=file.name,
                    use_container_width=True
                )
