from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")

import sys
import io

sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# load_docs.py ファイルの読み込み
from app.load_docs import load_contract
from app.vectorstore import create_vectorstore
from app.rag_chain import create_rag_chain
from app.user_question import user_question
#from app.followup import followup
#from app.node import question

from langchain_openai import ChatOpenAI
from langgraph.graph import Graph, END
from IPython.display import Image, display
from typing import TypedDict, List, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory

                             
documents = load_contract("./data/賃貸借契約書.txt")

# ドキュメントをチャンク化する　vectorstore.pyに引き渡す
vectorstore = create_vectorstore(documents, api_key)

class AgentState(TypedDict, total=False):
    #user_request: str  # ユーザーからの元の質問
    question: str  # ユーザーからの元の質問
    answer: str
    action: Literal["continue", "end"]
    end_message: str

# ======================================================================
# node 2026.2.9

# LLMの定義
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0,api_key=api_key)

def input_node(state: AgentState):
    #state["question"] = user_question("ユーザーからの質疑：")
    return {
        "question": state["question"]
    }

def question_node(state: AgentState):
    """
    賃貸借契約書からユーザーからの問いに該当する内容を抽出する
    """
    print("--- ノード：回答抽出 ---")

    #print("DEBUG question type:", type(state["user_request"]))
    #print("DEBUG question value:", state["user_request"])

# 検索機能（リトリーバー）の設定
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """あなたは賃貸借契約書の内容のみを根拠に回答するAIです。
            - 契約書に記載がない場合は「その内容に関しては契約書に記載がありませんので、お答えすることは出来ません。」と回答してください
            - 推測や一般論は禁止です
            - 回答には根拠となる条文番号を含めてください
            """
        ),
 #       ("placeholder","{history}"),
        ("human", "【契約書内容】\n{context}"),
        ("human", "{input}")
    ])


# 出力パーサーの設定
    output_parser = StrOutputParser()

# RAGパイプライン構築
    #chain = (
        #{"context": retriever, "input": RunnablePassthrough()}
        #{
        #"input": RunnableLambda(lambda x: x["input"]) | retriever,
        #"history": RunnableLambda(lambda x: x["history"]),
        #}
        #| prompt | llm | output_parser
    #)

    chain = ({
        "context": RunnableLambda(lambda x: x["question"]) | retriever,
        "input": RunnableLambda(lambda x: x["question"]),
        }
       | prompt 
       | llm 
       | output_parser
    )
    #chain = prompt | llm | output_parser

    memory = ConversationBufferMemory(return_messages=True)

    chat_runnable = RunnableWithMessageHistory(
        chain, # Corrected: use 'chain' instead of 'chat'
        lambda session_id: memory.chat_memory, # Corrected: use 'memory.chat_memory'
        input_messages_key="input", # Added missing comma
        history_messages_key="history"
    )

# 結果を取得
    #request_text = state["question"]
    answer = chain.invoke({
       "question": state["question"]      
       #"input":"家賃はいくら"
    })

#    print(f"-> ユーザーからの質問結果： {result}")

    return {
        "answer": answer
    }

def confirm_QA_node(state: AgentState):
    """
    質疑応答を終えるかどうかを確認する
    """
    print("--- ノード：質疑応答の終了確認 ---")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "あなたは対話の進行管理AIです。"
        "これ以上の質疑応答が必要かを判断してください。\n"
        "必要なら continue、十分なら end を返してください。"),
#        ("placeholder","{history}"),
        ("human", "{input}")
    ])

    chain = prompt | llm

    
# Streamlitでは絶対に使えません
#    user_input = user_question("質問事項はまだありますか？（はい or いいえ）：")

    result = chain.invoke({
        "input": user_input
    })

    print(result.content)

    return {
        "action": result.content
    }

def should_continue_node(state: AgentState):
    if state["action"] == "end":
        return "END"
    return "ASK"

def end_node(state: AgentState):
    """
    質疑応答を終了する旨を返す
    """
    print("--- ノード：質疑応答　終了")

    if state["action"] == "end":
        return {
            "end_message": "質疑応答を終わります。お疲れ様でした。"
        }


# ======================================================================
# Grapf

workflow = Graph()

workflow.add_node("input_node",input_node)
workflow.add_node("question_node",question_node)
#workflow.add_node("confirm_QA_node",confirm_QA_node)
#workflow.add_node("should_continue_node",should_continue_node)
#workflow.add_node("end_node",end_node)

workflow.set_entry_point("input_node")
workflow.add_edge("input_node", "question_node")
workflow.add_edge("question_node", END)

#workflow.add_edge("question_node", "confirm_QA_node")


#workflow.add_conditional_edges(
#    "confirm_QA_node",
#    should_continue_node,
#    {
#        "ASK": "input_node",
#        "END": "end_node"
#    }
#)
#workflow.add_edge("end_node",END)

app = workflow.compile()

#try:
#  display(Image(app.get_graph().draw_mermaid_png()))
#except Exception:
#  print("Graph visualization requires drawio-headless package. Skipping.")

def run_rag(question: str):
    result = app.invoke({"question": question})
    return result["answer"]

# -----------------------------------------------------
# Streamlit にて　表示させる

#print("---  賃貸借契約書の内容について、何か質問はありますか　---")
#result = app.invoke(
#    {
#        #"question": user_question("ユーザーからの質疑：")
#        #"user_request": "家賃はいくら？"
#    }
#)

#print(result)

#end_message = result.get("end_message")

#print({end_message})

# ----------------------------------------------------


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

# ======================================================================
# ２回まで回答 2026.2.9

# LLMに回答を求める　　　　rag_chain.pyに引き渡す
##chain = create_rag_chain(vectorstore, api_key)

# 質問内容をユーザーが入力
##user_questions = user_question("何か質問はありますか？:")

# 結果を取得
##result = chain.invoke(user_questions)

# 結果の表示
##print("【回答】")
##print(result)

# 類似質問の生成
##followup_chain = followup(api_key)

##followup_result = followup_chain.invoke({
##    "answer": result
##})

##print("\n【関連する質問】")
##print(followup_result)

# 質問内容をユーザーが入力
##user_questions = user_question("ほかに質問はありますか？:")

# 結果を取得
##result = chain.invoke(user_questions)

# 結果の表示
##print("【回答】")
##print(result)
# ======================================================================
