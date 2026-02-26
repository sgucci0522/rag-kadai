# RAG 課題（rag-kadai）

RAG（Retrieval-Augmented Generation）を用いた契約書QAシステムです。  
契約書データをベースに質問応答を行います。

## 🎯 概要

本システムは以下を目的としています。

- 契約書に対する質問応答（RAG）
- SQLite を用いたデータ管理
- Streamlit によるチャットUI
- 意図分類による処理分岐（RAG / メール生成）

## 🧱 構成
rag-kadai/
├── app/ # 生成モデル周り実装

├── data/ # データセット / ドキュメントなど

├── ui/ # Streamlit実装

├── poetry.lock # 依存パッケージのバージョンを固定

├── pyproject.toml # Pythonプロジェクトの設定ファイル

└── README.md # このファイル

## 🚀 セットアップ

### 1. Poetry インストール

未インストールの場合
bash
curl -sSL https://install.python-poetry.org | python3 -

### 2. 依存インストール
poetry install

### 3. 仮想環境に入る
poetry shell
または
poetry run <command>

### 3. 環境変数
.env を作成
OPENAI_API_KEY=xxxx

## 🚀 起動方法
poetry run streamlit run ui/chat.py
