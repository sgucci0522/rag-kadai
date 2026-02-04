from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#from langchain.chains import RetrievalQA
from .prompt import PROMPT

def create_rag_chain(vectorstore):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

# vectordb.as_retriever: 先ほど作成したChromaベクトルストアを「検索機能（リトリーバー）」として使える形に変換
    retriever = vectorstore.as_retriever(
        search_type="similarity",   #  ベクトル間の「類似度（similarity）」に基づいて検索を行う
        search_kwargs={"k": 3}      # 「上位3件の検索結果を返す」という指定
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False
    )

    return qa_chain
