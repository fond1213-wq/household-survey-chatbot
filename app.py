import streamlit as st

st.set_page_config(
    page_title="가구부문통계조사 챗봇",
    page_icon="📚",
    layout="wide"
)

# -----------------------------
# 제목
# -----------------------------
st.title("📚 가구부문통계조사 챗봇")

st.caption(
    "가구부문 통계조사 업무자료를 기반으로 질문에 답변하는 AI 챗봇"
)

st.divider()

# -----------------------------
# 사이드바
# -----------------------------
with st.sidebar:
    st.header("📂 자료 관리")

    menu = st.radio(
        "메뉴",
        [
            "질문하기",
            "자료관리"
        ]
    )

# -----------------------------
# 질문하기
# -----------------------------
if menu == "질문하기":

    st.subheader("💬 질문하기")

    question = st.text_area(
        "궁금한 내용을 입력하세요.",
        placeholder="예: 취업자는 어떤 기준으로 판단하나요?",
        height=120
    )

    if st.button("🔍 질문하기", use_container_width=True):

        if question.strip():
            st.info("AI 연결 전 테스트 화면입니다.")
            st.write("입력하신 질문:")
            st.write(question)

        else:
            st.warning("질문을 입력해주세요.")

# -----------------------------
# 자료관리
# -----------------------------
elif menu == "자료관리":

    st.subheader("📂 자료관리")

    st.write(
        "관리자만 PDF, TXT, 사진 자료를 추가할 수 있도록 "
        "구성할 예정입니다."
    )

    st.divider()

    st.write("📄 PDF 자료")
    st.file_uploader(
        "PDF 파일",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.write("📝 TXT 자료")
    st.file_uploader(
        "TXT 파일",
        type=["txt"],
        accept_multiple_files=True
    )

    st.write("📷 사진 자료")
    st.file_uploader(
        "사진 파일",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
