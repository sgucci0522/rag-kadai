from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_vectorstore(documents, api_key):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,		# 各チャンクの最大文字数を「800文字」に制限
        chunk_overlap=100,	# # 連続するチャンク間で200文字重複させる（文脈の連続性を保つため）
        separators=["\n\n", "\n", "。", "、", " ", ""]     # なるべく意味が途切れないように分けるために、テキストを分割する際に使う「区切り文字」のリスト
    )

    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=api_key
    )

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="tintai_contract" # contract 契約書
    )

    return vectorstore