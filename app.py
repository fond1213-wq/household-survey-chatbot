import streamlit as st

st.set_page_config(
    page_title="가구부문통계조사 챗봇",
    page_icon="📚"
)

st.title("📚 가구부문통계조사 챗봇")

st.write("가구부문 통계조사 업무자료를 검색하고 질문할 수 있는 챗봇입니다.")

question = st.text_input(
    "질문을 입력하세요",
    placeholder="예: 취업자는 어떻게 판단하나요?"
)

if st.button("질문하기"):
    if question:
        st.info("현재는 테스트 단계입니다.")
        st.write("질문:", question)
    else:
        st.warning("질문을 입력해주세요.")
