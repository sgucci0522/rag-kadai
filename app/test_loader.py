from load_docs import load_contract

docs = load_contract("../data/賃貸借契約書.txt")
print(docs[0].page_content[:500])
