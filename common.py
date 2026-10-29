"""投入スクリプトと検索関数で共通に使う処理（正規化・言語判定・単語分割・ベクトル化・DB接続）。

正規化ロジックは「投入時」と「検索時」で必ず同じものを使うこと。
ずれると完全一致判定（*_hash）やトライグラム類似度が正しく出ない。
"""
from __future__ import annotations

import os
import re
import unicodedata
from functools import lru_cache
from typing import Literal, Sequence

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

Lang = Literal["en", "ja"]
LANGS: tuple[Lang, ...] = ("en", "ja")
LANG_LABEL = {"en": "英文", "ja": "和文"}

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
    "–": "-", "—": "-", "−": "-", " ": " ",
    "〜": "~",  # 〜（波ダッシュ）→ ~（NFKC 後の全角チルダと統一）
})
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """表記ゆれだけを整える軽い正規化（大文字小文字・空白は保持）。ベクトル化・差分表示に使う。"""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(_QUOTE_MAP)
    return _WS.sub(" ", text).strip()


def normalize_for_match(text: str, lang: Lang) -> str:
    """トライグラム検索・完全一致判定用の正規化。

    英文: 小文字化（空白は1つにまとめる）
    和文: 全角半角統一（NFKC）・空白をすべて除去・英字は小文字化
    """
    text = clean_text(text).lower()
    if lang == "ja":
        text = _WS.sub("", text)
    return text


# ---------------------------------------------------------------------
# 言語判定
# ---------------------------------------------------------------------
_JA_CHARS = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]")


def detect_lang(text: str) -> Lang:
    """かな・漢字を含めば和文、それ以外は英文とみなす。"""
    return "ja" if _JA_CHARS.search(text or "") else "en"


# ---------------------------------------------------------------------
# 単語分割（差分表示・一致率用）
# ---------------------------------------------------------------------
_EN_TOKEN = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]", re.UNICODE)
# 形態素解析が使えない場合の簡易分割（文字種の切れ目で区切る）
_JA_FALLBACK = re.compile(
    r"[一-鿿㐀-䶿々〆ヵヶ]+|[぀-ゟ]+|[゠-ヿー]+"
    r"|[A-Za-z0-9]+(?:[.,][0-9]+)*%?|\S"
)
_ASCII_PIECE = re.compile(r"^[A-Za-z0-9.,%]+$")


@lru_cache(maxsize=1)
def _ja_tagger():
    try:
        import fugashi  # 形態素解析（辞書: unidic-lite）
        return fugashi.Tagger()
    except Exception:  # 未インストール・辞書なし
        return None


def tokenize(text: str, lang: Lang) -> list[str]:
    """差分表示用の単語分割。英文は単語、和文は形態素（使えなければ文字種）単位。"""
    text = clean_text(text)
    if lang == "en":
        return _EN_TOKEN.findall(text)

    tagger = _ja_tagger()
    if tagger is None:
        return _JA_FALLBACK.findall(text)
    tokens: list[str] = []
    for word in tagger(text):
        surface = word.surface
        # 「12 . 5 %」「CO 2」のように分かれた英数字は、間に空白がなければ1語にまとめる
        if (tokens and not word.white_space and _ASCII_PIECE.match(surface)
                and _ASCII_PIECE.match(tokens[-1])):
            tokens[-1] += surface
        else:
            if word.white_space and tokens:
                tokens.append(" ")  # 和文中の空白も差分表示で再現する
            tokens.append(surface)
    return tokens


# ---------------------------------------------------------------------
# ベクトル化
# ---------------------------------------------------------------------
class Embedder:
    """sentence-transformers によるベクトル化（multilingual-e5 は日英両対応）。

    e5 系モデルは入力に接頭辞が必要。今回は「英文 ↔ 英文」「和文 ↔ 和文」の
    対称な類似検索なので、投入時・検索時とも "query: " をそろえて付ける（e5 の推奨用法）。
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
