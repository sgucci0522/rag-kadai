# load_docs.py ファイルの読み込み
from app.load_docs import load_contract
from app.vectorstore import create_vectorstore
from app.rag_chain import create_rag_chain
from app.user_question import user_question
from app.followup import followup
from app.node import question

from langgraph.graph import Graph, END
from IPython.display import Image, display

import os
api_key = os.getenv("OPENAI_API_KEY")

documents = load_contract("./data/賃貸借契約書.txt")

# ドキュメントをチャンク化する　vectorstore.pyに引き渡す
vectorstore = create_vectorstore(documents, api_key)

# 一旦、動ける状態まで戻す

#def question(state):
#    api_key = state["api_key"]
#    input = state["input"]

#workflow = Graph()

#workflow.add_node("question",question)

#workflow.set_entry_point("question")

# グラフをコンパイル
#app = workflow.compile()

#input ={"input":"何か質問はありますか？:"}
#result = app.invoke({
#    "input": input,
#    "api_key": api_key
#})

#print(result["response"])

# グラフ構造を可視化
#try:
#    display(Image(app.get_graph().draw_mermaid_png()))
#except Exception:
#    print("Graph visualization requires drawio-headless package. Skipping.")


# LLMに回答を求める　　　　rag_chain.pyに引き渡す
chain = create_rag_chain(vectorstore, api_key)

# 質問内容をユーザーが入力
user_questions = user_question("何か質問はありますか？:")

# 結果を取得
result = chain.invoke(user_questions)

# 結果の表示
print("【回答】")
print(result)

# 類似質問の生成
followup_chain = followup(api_key)

followup_result = followup_chain.invoke({
    "answer": result
})

print("\n【関連する質問】")
print(followup_result)

# 質問内容をユーザーが入力
user_questions = user_question("ほかに質問はありますか？:")

# 結果を取得
result = chain.invoke(user_questions)

# 結果の表示
print("【回答】")
print(result)
