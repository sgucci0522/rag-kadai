# load_docs.py ファイルの読み込み
from app.load_docs import load_contract
from app.vectorstore import create_vectorstore
from app.rag_chain import create_rag_chain

from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
api_key = os.getenv("OPENAI_API_KEY")

#from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#from langchain_community.vectorstores import Chroma
#from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.output_parsers import StrOutputParser

#from langchain_core.runnables import RunnablePassthrough


documents = load_contract("./data/賃貸借契約書.txt")
print(documents[0].page_content[:500])

# vectorstore.pyに引き渡す

vectorstore = create_vectorstore(documents, api_key)

# --------------------------------------------------------------------
# ドキュメントのチャンク化
#recursive_splitter = RecursiveCharacterTextSplitter(
#    chunk_size=1000,        # 各チャンクの最大文字数を「1000文字」に制限
#    chunk_overlap=200,      # 連続するチャンク間で200文字重複させる（文脈の連続性を保つため）
#    separators=["\n\n", "\n", "。", "、", " ", ""]     # なるべく意味が途切れないように分けるために、テキストを分割する際に使う「区切り文字」のリスト
#)
#recursive_docs = recursive_splitter.split_documents(documents)
#docs = recursive_docs
#
# チャンク数の確認
#print(len(docs))
#
# 埋め込みモデルの設定　取得・分割したテキストを「ベクトル（数値の並び）」に変換
#embeddings = OpenAIEmbeddings(
#    model="text-embedding-ada-002",
#    api_key=api_key
#)
#
# ベクトルストアの作成　チャンク化されたドキュメントを一括でベクトル化しDBに保存
#vectordb = Chroma.from_documents(
#    documents=docs,
#    embedding=embeddings,
#    collection_name="tintai_contract" # contract 契約書
#)
# --------------------------------------------------------------------


# データベースに格納されたドキュメント数を確認
count = vectorstore._collection.count()
print(count)


# rag_chain.pyに引き渡す

# 検索機能（リトリーバー）の設定
#retriever = vectordb.as_retriever(
#    search_type="similarity",
#    search_kwargs={"k": 3}
#)
#
# 言語モデル（LLM）の初期化
#llm = ChatOpenAI(
#    model_name="gpt-4o-mini",
#    temperature=0,
#    api_key=api_key
#)
#
# プロンプトテンプレートの作成
#prompt = ChatPromptTemplate.from_messages([
#    ("system", "与えられた質問に対して、以下のコンテキストを使用して回答してください。コンテキスト:{context}"),
#    ("human", "{input}")
#])

# 出力パーサーの設定
#output_parser = StrOutputParser()

# RAGパイプライン構築
#chain = (
#    {"context": retriever, "input": RunnablePassthrough()}
#    | prompt | llm | output_parser
#)

chain = create_rag_chain(vectorstore, api_key)

# 質問を設定
#query = "一ヶ月の家賃を教えて"
query = "解約したいときは、どうすればいい？いつまでに連絡しないといけない？"

# 結果を取得
result = chain.invoke(query)

# 結果の表示
print(result)
