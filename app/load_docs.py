from pathlib import Path
from langchain_community.document_loaders import (
    Docx2txtLoader,
    TextLoader
)
import sqlite3
from langchain.schema import Document

def load_contract(path: str):
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"{path} が見つかりません")

    if path.suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    elif path.suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")

    else:
        raise ValueError("対応していないファイル形式です")

    return loader.load()
