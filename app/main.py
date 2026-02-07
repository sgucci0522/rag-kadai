# load_docs.py ファイルの読み込み
from app.load_docs import load_contract
from app.vectorstore import create_vectorstore
from app.rag_chain import create_rag_chain
from app.user_question import user_question

import os
api_key = os.getenv("OPENAI_API_KEY")

documents = load_contract("./data/賃貸借契約書.txt")

# ドキュメントをチャンク化する　vectorstore.pyに引き渡す
vectorstore = create_vectorstore(documents, api_key)

# LLMに回答を求める　　　　rag_chain.pyに引き渡す
chain = create_rag_chain(vectorstore, api_key)

# 質問内容をユーザーが入力
user_questions = user_question()

# 結果を取得
result = chain.invoke(user_questions)

# 結果の表示
print(result)
