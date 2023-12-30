from dotenv import load_dotenv
load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

import streamlit as st

a = "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"

# llm = OpenAI(openai_api_key=a)
# result = llm.predict("안녕")
# print(result)
chat_model = ChatOpenAI(openai_api_key=a)
st.title("내가 나는솔로에 간다면?!")
story = st.text_input("에시 상황을 제시해주세요.")
prompt = "에 대한 대응방안을 작성해줘. 이때 이모티콘도 써주고, 나를 연애에 목마른 사람입장에서 작성해줘. 이때 3가지로 방안을 제시해주면 고마울 것 같아."

if st.button("가보자 솔로나라"):
    st.write("나는", story, "어떻게 해야할까? 도와줘 AI 선생")
    with st.spinner("30초 후에 공개됩니다."):
        result = chat_model.predict(story + prompt)
        st.balloons()
        st.write(result)