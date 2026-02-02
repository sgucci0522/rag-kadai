from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from .prompt import PROMPT

def create_rag_chain(vectorstore):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False
    )

    return qa_chain
