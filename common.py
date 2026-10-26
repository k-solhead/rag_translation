"""投入スクリプトと検索関数で共通に使う処理（正規化・ベクトル化・DB接続）。
 
正規化ロジックは「投入時」と「検索時」で必ず同じものを使うこと。
ずれると完全一致判定（source_hash）やトライグラム類似度が正しく出ない。
"""
from __future__ import annotations
 
import os
import re
import unicodedata
from typing import Sequence
 
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
 
# ---------------------------------------------------------------------
# 設定（環境変数で上書き可能）
# ---------------------------------------------------------------------
DATABASE_URL = os.getenv("TM_DATABASE_URL", "postgresql://tm:tm@localhost:5432/tm")
EMBED_MODEL = os.getenv("TM_EMBED_MODEL", "intfloat/multilingual-e5-small")
EMBED_DIM = int(os.getenv("TM_EMBED_DIM", "384"))  # schema.sql の vector(n) と一致させる
 
# ---------------------------------------------------------------------
# 正規化
# ---------------------------------------------------------------------
_QUOTE_MAP = str.maketrans({
    "‘": "'", "’": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "‟": '"', "″": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
})
_WS = re.compile(r"\s+")
 
 
def clean_text(text: str) -> str:
    """表記ゆれだけを整える軽い正規化（大文字小文字は保持）。ベクトル化に使う。"""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(_QUOTE_MAP)
    return _WS.sub(" ", text).strip()
 
 
def normalize_for_match(text: str) -> str:
    """トライグラム検索・完全一致判定用の正規化（小文字化まで行う）。"""
    return clean_text(text).lower()
 
 
# ---------------------------------------------------------------------
# ベクトル化
# ---------------------------------------------------------------------
class Embedder:
    """sentence-transformers によるベクトル化。
 
    e5 系モデルは入力に接頭辞が必要。今回は「英文 ↔ 英文」の対称な類似検索なので、
    投入時・検索時とも "query: " をそろえて付ける（e5 の推奨用法）。
    """
 
    def __init__(self, model_name: str = EMBED_MODEL, device: str | None = None):
        from sentence_transformers import SentenceTransformer  # 起動時間短縮のため遅延 import
 
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.prefix = "query: " if "e5" in model_name.lower() else ""
 
    def encode(self, texts: Sequence[str], batch_size: int = 64,
               show_progress: bool = False) -> np.ndarray:
        inputs = [self.prefix + clean_text(t) for t in texts]
        vecs = self.model.encode(
            inputs,
            batch_size=batch_size,
            normalize_embeddings=True,   # コサイン類似度用に正規化
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return vecs.astype(np.float32)
 
 
# ---------------------------------------------------------------------
# DB 接続
# ---------------------------------------------------------------------
def connect(dsn: str = DATABASE_URL) -> psycopg.Connection:
    conn = psycopg.connect(dsn)
    register_vector(conn)  # numpy 配列 ⇔ vector 型を自動変換
    return conn
