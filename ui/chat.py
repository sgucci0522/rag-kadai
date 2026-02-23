import sys
from pathlib import Path

# 現在のファイルの場所を絶対パスに変換　.parent 一つ上のフォルダ
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from app.main import app  # あなたのRAGチェーン
from app.main import run_rag
from app.main import classify_intent
from app.main import generate_landlord_mail
from app.main import database_search

st.title("📄 RAG Chat System")

if "messages" not in st.session_state:
    st.session_state.messages = []

# チャット履歴表示　session_state ブラウザセッション単位の保存領域
# 過去の会話を１件ずつ取り出す
for msg in st.session_state.messages:

# roleに応じて　"user"-> ユーザー　"assistant"->ＡＩ
    with st.chat_message(msg["role"]):

# メッセージ内容を表示
        st.markdown(msg["content"])

# 入力
if prompt := st.chat_input("質問を入力してください"):

# ユーザーの入力内容を履歴に追加、吹き出しを表示
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG実行
    #result = app.invoke({"question": prompt})

    intent = classify_intent({"question": prompt})
    intent = intent["classify"]

    print(f"intent: {intent}")

    if intent == "1":
        answer = run_rag(prompt)
    elif intent == "2":
        result = generate_landlord_mail({"question": prompt})
# RAGの戻り値から回答部分だけ取り出す
        answer = result["answer"]
    else:
        result = database_search({"question": prompt})
        answer = result["answer"]

# ＡＩの返答を履歴に保存、吹き出しを表示
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)
