from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

#from langchain.chains import RetrievalQA
#from .old_prompt import PROMPT

def create_rag_chain(vectorstore, api_key):

# 言語モデル（LLM）の初期化
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key
    )

# 検索機能（リトリーバー）の設定
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

# プロンプトテンプレートの作成
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """あなたは賃貸借契約書の内容のみを根拠に回答するAIです。
    - 契約書に記載がない場合は「その内容に関しては分かり兼ねます」と回答してください
    - 推測や一般論は禁止です
    - 回答には根拠となる条文番号を含めてください
    """
        ),
        ("placeholder","{history}"),
        ("human", "【契約書内容】\n{context}"),
        ("human", "【質問】\n{input}")
    ])
   # prompt = ChatPromptTemplate.from_messages([
   #     ("system", "与えられた質問に対して、以下のコンテキストを使用して回答してください。コンテキスト:{context}"),
   #     ("human", "{input}")
   # ])

# 出力パーサーの設定
    output_parser = StrOutputParser()

# RAGパイプライン構築
    chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt | llm | output_parser
    )

    return chain
