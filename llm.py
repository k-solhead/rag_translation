"""LLM への接続を共通化する（LM Studio のローカル LLM と Claude API）。

翻訳エンジン（Engine）は「どこに接続するか（provider）」と
「どの形式でプロンプトを渡すか（style）」の組み合わせで表す。

    provider   style            説明
    lmstudio   rag              チャット形式。RAG（前年訳・差分・用語集）を使うプロンプト
    lmstudio   plain            チャット形式。従来の translate と同じ短いプロンプト（ファインチューニング済みモデル用）
    lmstudio   plamo            plamo-2-translate 専用（テキスト補完 API。<|plamo:op|> 形式で前年の対訳を few-shot として渡す）
    lmstudio   translategemma   translategemma 専用（テキスト補完 API。修正モードは前年訳のポストエディットを指示する）
    anthropic  rag              Claude API。RAG を使うプロンプト

LM Studio のモデルは、モデル名から style を自動判定する（detect_style）。

環境変数
    LMSTUDIO_BASE_URL     LM Studio の接続先（既定: http://192.168.11.83:1234/v1）
    LMSTUDIO_CONCURRENCY  LM Studio への同時リクエスト数（既定: 1）
    ANTHROPIC_API_KEY     Claude API のキー（未設定なら Claude は選択肢に出ない）
    CLAUDE_MODELS         選択肢に出す Claude のモデル（カンマ区切り。既定: claude-haiku-5-5,claude-sonnet-5-5）
    CLAUDE_CONCURRENCY    Claude API への同時リクエスト数（既定: 4）
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

import requests

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://192.168.11.83:1234/v1").rstrip("/")
LMSTUDIO_CONCURRENCY = int(os.getenv("LMSTUDIO_CONCURRENCY", "1"))
CLAUDE_MODELS = [m.strip() for m in os.getenv(
    "CLAUDE_MODELS", "claude-haiku-5-5,claude-sonnet-5-5").split(",") if m.strip()]
CLAUDE_CONCURRENCY = int(os.getenv("CLAUDE_CONCURRENCY", "4"))
REQUEST_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))

STYLE_LABEL = {
    "rag": "RAG",
    "plain": "従来プロンプト",
    "plamo": "PLaMo翻訳",
    "translategemma": "TranslateGemma",
}


@dataclass(frozen=True)
class Engine:
    provider: str   # "lmstudio" / "anthropic"
    model: str
    style: str      # "rag" / "plain" / "plamo" / "translategemma"

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}:{self.style}"

    @property
    def label(self) -> str:
        where = "Claude" if self.provider == "anthropic" else "LM Studio"
        return f"{self.model}（{where}・{STYLE_LABEL[self.style]}）"

    @property
    def uses_rag(self) -> bool:
        return self.style == "rag"

    @property
    def concurrency(self) -> int:
        return CLAUDE_CONCURRENCY if self.provider == "anthropic" else LMSTUDIO_CONCURRENCY


def detect_style(model: str) -> str:
    """LM Studio のモデル名からプロンプト形式を判定する。"""
    name = model.lower()
    if "plamo" in name and "translate" in name:
        return "plamo"
    if "translategemma" in name:
        return "translategemma"
    if name.startswith("fine_tuned"):
        return "plain"
    return "rag"


# ---------------------------------------------------------------------
# 利用できるエンジンの一覧
# ---------------------------------------------------------------------
def fetch_lmstudio_models() -> list[str] | None:
    """LM Studio（OpenAI 互換 API）から読み込み可能なモデル一覧を取得する。接続できなければ None。"""
    try:
        r = requests.get(f"{LMSTUDIO_BASE_URL}/models", timeout=5)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])
                if "embed" not in m["id"].lower()]  # 埋め込みモデルは除外
    except Exception:
        return None


def claude_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def available_engines() -> tuple[list[Engine], list[str]]:
    """(使えるエンジン, 警告メッセージ) を返す。"""
    engines: list[Engine] = []
    warnings: list[str] = []
    models = fetch_lmstudio_models()
    if models is None:
        warnings.append(f"LM Studio（{LMSTUDIO_BASE_URL}）に接続できません。ローカル LLM は使えません。")
    else:
        for m in models:
            style = detect_style(m)
            engines.append(Engine("lmstudio", m, style))
            if style == "plain":  # ファインチューニング済みモデルは RAG プロンプトでも試せるようにする
                engines.append(Engine("lmstudio", m, "rag"))
    if claude_available():
        engines += [Engine("anthropic", m, "rag") for m in CLAUDE_MODELS]
    else:
        warnings.append("ANTHROPIC_API_KEY が未設定のため、Claude は使えません（.env に設定してください）。")
    return engines, warnings


# ---------------------------------------------------------------------
# 呼び出し
# ---------------------------------------------------------------------
@lru_cache(maxsize=1)
def _openai_client():
    import openai
    return openai.OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio",
                         timeout=REQUEST_TIMEOUT, max_retries=2)


@lru_cache(maxsize=1)
def _anthropic_client():
    import anthropic
    return anthropic.Anthropic(timeout=REQUEST_TIMEOUT, max_retries=4)


_WRAP_TAGS = re.compile(r"^\s*<(translation|output)>\s*(.*?)\s*</\1>\s*$", re.S)
_QUOTES = [("「", "」"), ('"', '"'), ("'", "'"), ("“", "”")]


def clean_output(text: str) -> str:
    """LLM の出力から、前後の空白・タグ・全体を囲む引用符を取り除く。"""
    text = (text or "").strip()
    m = _WRAP_TAGS.match(text)
    if m:
        text = m.group(2).strip()
    for left, right in _QUOTES:
        if len(text) > 2 and text.startswith(left) and text.endswith(right) and text.count(left) == 1:
            text = text[1:-1].strip()
    return text


def chat(engine: Engine, system: str, user: str, *, temperature: float = 0.1,
         max_tokens: int = 2048) -> str:
    """チャット形式で1回呼び出す（LM Studio / Claude）。"""
    if engine.provider == "anthropic":
        kwargs = {}
        if os.getenv("CLAUDE_TEMPERATURE"):
            kwargs["temperature"] = float(os.environ["CLAUDE_TEMPERATURE"])
        msg = _anthropic_client().messages.create(
            model=engine.model,
            max_tokens=max_tokens,
            # システムプロンプトは全文共通なのでキャッシュする（費用と応答時間の削減）
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            **kwargs,
        )
        return clean_output("".join(b.text for b in msg.content if b.type == "text"))

    res = _openai_client().chat.completions.create(
        model=engine.model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return clean_output(res.choices[0].message.content)


def complete(engine: Engine, prompt: str, *, stop: list[str] | None = None,
             max_tokens: int = 2048) -> str:
    """テキスト補完形式で1回呼び出す（plamo-2-translate / translategemma 用。LM Studio のみ）。"""
    res = _openai_client().completions.create(
        model=engine.model, prompt=prompt, stop=stop, max_tokens=max_tokens, temperature=0,
    )
    return clean_output(res.choices[0].text)
