from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")

import sys
import io
import sqlite3

sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# load_docs.py ファイルの読み込み
from app.load_docs import load_contract
from app.vectorstore import create_vectorstore
#from app.rag_chain import create_rag_chain
#from app.user_question import user_question

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

from langchain.schema import Document

documents = []

documents.append(
    Document(
        page_content="##### 賃貸借契約書 #####",
        metadata={"type": "contract"}
    )
)

documents.extend(load_contract("./data/賃貸借契約書.txt"))

#データベースをテキストで取り込む
conn = sqlite3.connect("./data/app.db")
cur = conn.cursor()

rows = cur.execute("SELECT target_date, amount, status FROM rent_payments").fetchall()

docs = []
for target_date, amount, status in rows:
    text = f"年月:{target_date} 家賃支払い:{amount} 支払い状態:{status}"
    docs.append(Document(page_content=text))

documents.append(
    Document(
        page_content="##### 支払い情報 #####",
        metadata={"type": "payment",
                  "date": str(target_date),
                  "status": status
                  }
        )
)

documents.extend(docs)

# ドキュメントをチャンク化する　vectorstore.pyに引き渡す
vectorstore = create_vectorstore(documents, api_key)

class AgentState(TypedDict, total=False):
    #user_request: str  # ユーザーからの元の質問
    question: str  # ユーザーからの元の質問
    answer: str
    classift: str
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

    chain = ({
        "context": RunnableLambda(lambda x: x["question"]) | retriever,
        "input": RunnableLambda(lambda x: x["question"]),
        }
       | prompt 
       | llm 
       | output_parser
    )

    memory = ConversationBufferMemory(return_messages=True)

    chat_runnable = RunnableWithMessageHistory(
        chain, # Corrected: use 'chain' instead of 'chat'
        lambda session_id: memory.chat_memory, # Corrected: use 'memory.chat_memory'
        input_messages_key="input", # Added missing comma
        history_messages_key="history"
    )

# 結果を取得
    answer = chain.invoke({
       "question": state["question"]      

    })

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

    #chain = prompt | llm
    
    chain = ({
        "input": RunnableLambda(lambda x: x["question"]),
        }
       | prompt 
       | llm 
       | output_parser
    )

    answer = chain.invoke({
        "question": state["question"]
    })

    return {
        "answer": answer
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

def generate_landlord_mail(state: AgentState):
    """
    貸主に対するメール本文を返す
    """
# 検索機能（リトリーバー）の設定
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "あなたは不動産管理の専門家です。"
         "次の賃貸契約書を参照し、"
         "貸主（甲）の氏名を特定したうえで、"
         "貸主宛の丁寧なメール本文を作成してください。"
         "【条件】"
         " ・メール本文に貸主名を必ず含める"
         " ・丁寧語"
         " ・ビジネス文"
         " ・件名は不要"
         " ・署名は不要"
         " ・貸主名が不明な場合は「貸主様」とする"
         ),
        ("human", "【契約書内容】\n{context}"),
        ("human", "{input}")
    ])

    #chain = prompt | llm

# 出力パーサーの設定
    output_parser = StrOutputParser()

    chain = ({
        "context": RunnableLambda(lambda x: x["question"]) | retriever,
        "input": RunnableLambda(lambda x: x["question"]),
        }
       | prompt 
       | llm 
       | output_parser
    )

# 結果を取得
    answer = chain.invoke({
       "question": state["question"]

    })
    
    return {
        "answer": answer
    }


def classify_intent(state: AgentState):

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
あなたは意図分類AIです。
次の文章がどれに該当するか番号だけで答えてください。

1 = 通常の質問
2 = 貸主へのメール作成依頼

文章:
{input}
""")

# 出力パーサーの設定
    output_parser = StrOutputParser()
    #chain = prompt | llm
    
    chain = ({
        "input": RunnableLambda(lambda x: x["question"]),
        }
       | prompt 
       | llm 
       | output_parser
    )

    result = chain.invoke({
        #"input": user_input})
        "question": state["question"]
        })
   
    print(f"classify: {result}")

    return {
        "classify": result
    }
    #return result.strip()

def run_rag(question: str):
    result = app.invoke({"question": question})
    return result["answer"]

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

app = workflow.compile()

#try:
#  display(Image(app.get_graph().draw_mermaid_png()))
#except Exception:
#  print("Graph visualization requires drawio-headless package. Skipping.")

