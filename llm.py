"""LLM への接続を共通化する（LM Studio / Ollama のローカル LLM と、Claude・OpenAI・Gemini の API）。

翻訳エンジン（Engine）は「どこに接続するか（provider）」と
「どの形式でプロンプトを渡すか（style）」の組み合わせで表す。

    provider   style            説明
    lmstudio   rag              チャット形式。RAG（前年訳・差分・用語集）を使うプロンプト
    lmstudio   plain            チャット形式。従来の translate と同じ短いプロンプト（ファインチューニング済みモデル用）
    lmstudio   plamo            plamo-2-translate 専用（テキスト補完 API。<|plamo:op|> 形式で前年の対訳を few-shot として渡す）
    lmstudio   translategemma   translategemma 専用（テキスト補完 API。修正モードは前年訳のポストエディットを指示する）
    ollama     (同上)           LM Studio と同じ 4 種類。Ollama のネイティブ API（/api/chat・/api/generate）で呼ぶ
    anthropic  rag              Claude API。RAG を使うプロンプト
    openai     rag              OpenAI API。RAG を使うプロンプト
    gemini     rag              Gemini API（Google 公式の OpenAI 互換エンドポイント）。RAG を使うプロンプト

LM Studio / Ollama のモデルは、モデル名から style を自動判定する（detect_style）。

環境変数
    LMSTUDIO_BASE_URL     LM Studio の接続先（既定: http://192.168.11.83:1234/v1）
    LMSTUDIO_CONCURRENCY  LM Studio への同時リクエスト数（既定: 1）
    OLLAMA_BASE_URL       Ollama の接続先（既定: http://192.168.11.83:11434。空にすると Ollama は使わない）
    OLLAMA_CONCURRENCY    Ollama への同時リクエスト数（既定: 1）
    OLLAMA_NUM_CTX        Ollama のコンテキスト長（既定: 8192。Ollama 既定のままだと RAG プロンプトが切り詰められる）
    ANTHROPIC_API_KEY     Claude API のキー（未設定なら Claude は選択肢に出ない）
    CLAUDE_MODELS         選択肢に出す Claude のモデル（カンマ区切り。既定: claude-haiku-5-5,claude-sonnet-5-5）
    CLAUDE_CONCURRENCY    Claude API への同時リクエスト数（既定: 4）
    OPENAI_API_KEY        OpenAI API のキー（未設定なら OpenAI は選択肢に出ない）
    OPENAI_MODELS         選択肢に出す OpenAI のモデル（カンマ区切り。既定: gpt-6-luna,gpt-6.1-sol）
    OPENAI_CONCURRENCY    OpenAI API への同時リクエスト数（既定: 4）
    GEMINI_API_KEY        Gemini API のキー（未設定なら Gemini は選択肢に出ない）
    GEMINI_MODELS         選択肢に出す Gemini のモデル（カンマ区切り。既定: gemini-3.8-flash）
    GEMINI_CONCURRENCY    Gemini API への同時リクエスト数（既定: 4）
    OPENAI_REASONING_EFFORT / GEMINI_REASONING_EFFORT
                          推論の深さ（任意。例: low。空なら各社の既定）
    OPENAI_TEMPERATURE / GEMINI_TEMPERATURE
                          temperature（任意。指定できないモデルがあるため、空なら送らない）
    CLOUD_MAX_TOKENS      OpenAI / Gemini の出力トークン上限（既定: 8192。推論型モデルは推論にも使うため多めにする）
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

import requests

from usage import record_anthropic, record_openai_compatible

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://192.168.11.83:1234/v1").rstrip("/")
LMSTUDIO_CONCURRENCY = int(os.getenv("LMSTUDIO_CONCURRENCY", "1"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.11.83:11434").rstrip("/")
OLLAMA_CONCURRENCY = int(os.getenv("OLLAMA_CONCURRENCY", "1"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
CLAUDE_MODELS = [m.strip() for m in os.getenv(
    "CLAUDE_MODELS", "claude-haiku-5-5,claude-sonnet-5-5").split(",") if m.strip()]
CLAUDE_CONCURRENCY = int(os.getenv("CLAUDE_CONCURRENCY", "4"))


def _models(name: str, default: str) -> list[str]:
    return [m.strip() for m in os.getenv(name, default).split(",") if m.strip()]


OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_MODELS = _models("OPENAI_MODELS", "gpt-6-luna,gpt-6.1-sol")
OPENAI_CONCURRENCY = int(os.getenv("OPENAI_CONCURRENCY", "4"))
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL",
                            "https://generativelanguage.googleapis.com/v1beta/openai/").rstrip("/") + "/"
GEMINI_MODELS = _models("GEMINI_MODELS", "gemini-3.8-flash")
GEMINI_CONCURRENCY = int(os.getenv("GEMINI_CONCURRENCY", "4"))
CLOUD_MAX_TOKENS = int(os.getenv("CLOUD_MAX_TOKENS", "8192"))
REQUEST_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))

STYLE_LABEL = {
    "rag": "RAG",
    "plain": "従来プロンプト",
    "plamo": "PLaMo翻訳",
    "translategemma": "TranslateGemma",
}
PROVIDER_LABEL = {"lmstudio": "LM Studio", "ollama": "Ollama", "anthropic": "Claude",
                  "openai": "OpenAI", "gemini": "Gemini"}
# API キーで使う外部サービス: provider → (表示名, キーの環境変数, モデル一覧)
CLOUD_PROVIDERS = {
    "anthropic": ("Claude", "ANTHROPIC_API_KEY", CLAUDE_MODELS),
    "openai": ("OpenAI", "OPENAI_API_KEY", OPENAI_MODELS),
    "gemini": ("Gemini", "GEMINI_API_KEY", GEMINI_MODELS),
}


@dataclass(frozen=True)
class Engine:
    provider: str   # "lmstudio" / "ollama" / "anthropic" / "openai" / "gemini"
    model: str
    style: str      # "rag" / "plain" / "plamo" / "translategemma"

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}:{self.style}"

    @property
    def label(self) -> str:
        return f"{self.model}（{PROVIDER_LABEL[self.provider]}・{STYLE_LABEL[self.style]}）"

    @property
    def uses_rag(self) -> bool:
        return self.style == "rag"

    @property
    def concurrency(self) -> int:
        return {"anthropic": CLAUDE_CONCURRENCY, "ollama": OLLAMA_CONCURRENCY,
                "openai": OPENAI_CONCURRENCY, "gemini": GEMINI_CONCURRENCY}.get(
            self.provider, LMSTUDIO_CONCURRENCY)


def detect_style(model: str) -> str:
    """LM Studio / Ollama のモデル名からプロンプト形式を判定する。"""
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


def fetch_ollama_models() -> list[str] | None:
    """Ollama からインストール済みモデル一覧を取得する。接続できなければ None。"""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])
                if "embed" not in m["name"].lower()  # 埋め込みモデルは除外
                and not m.get("remote_host")]        # クラウドモデル（:cloud）は原稿が外部に送られるので除外
    except Exception:
        return None


def _local_engines(provider: str, models: list[str]) -> list[Engine]:
    engines: list[Engine] = []
    for m in models:
        style = detect_style(m)
        engines.append(Engine(provider, m, style))
        if style == "plain":  # ファインチューニング済みモデルは RAG プロンプトでも試せるようにする
            engines.append(Engine(provider, m, "rag"))
    return engines


def claude_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def provider_available(provider: str) -> bool:
    return bool(os.getenv(CLOUD_PROVIDERS[provider][1]))


def available_engines() -> tuple[list[Engine], list[str]]:
    """(使えるエンジン, 警告メッセージ) を返す。"""
    engines: list[Engine] = []
    warnings: list[str] = []
    # ---- ローカル LLM（LM Studio / Ollama）への接続は、現在使わないためコメントアウト ----
    # 再び使う場合は、以下のコメントを外してください（接続先は .env の LMSTUDIO_BASE_URL / OLLAMA_BASE_URL）。
    # models = fetch_lmstudio_models()
    # if models is None:
    #     warnings.append(f"LM Studio（{LMSTUDIO_BASE_URL}）に接続できません。ローカル LLM は使えません。")
    # else:
    #     engines += _local_engines("lmstudio", models)
    # if OLLAMA_BASE_URL:
    #     models = fetch_ollama_models()
    #     if models is None:
    #         warnings.append(f"Ollama（{OLLAMA_BASE_URL}）に接続できません。Ollama のモデルは使えません。")
    #     else:
    #         engines += _local_engines("ollama", models)
    missing = []
    for provider, (name, key_var, models) in CLOUD_PROVIDERS.items():
        if provider_available(provider):
            engines += [Engine(provider, m, "rag") for m in models]
        else:
            missing.append(f"{name}（{key_var}）")
    if missing:
        warnings.append("API キーが未設定のため使えません: " + "、".join(missing)
                        + "。使う場合は .env に設定してください。")
    return engines, warnings


# ---------------------------------------------------------------------
# 呼び出し
# ---------------------------------------------------------------------
@lru_cache(maxsize=1)
def _openai_client():
    import openai
    return openai.OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio",
                         timeout=REQUEST_TIMEOUT, max_retries=2)


@lru_cache(maxsize=2)
def _cloud_openai_client(provider: str):
    """OpenAI / Gemini 用のクライアント（Gemini は Google 公式の OpenAI 互換エンドポイント）。"""
    import openai
    base_url = OPENAI_BASE_URL if provider == "openai" else GEMINI_BASE_URL
    return openai.OpenAI(base_url=base_url, api_key=os.environ[CLOUD_PROVIDERS[provider][1]],
                         timeout=REQUEST_TIMEOUT, max_retries=4)


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


def _ollama(path: str, payload: dict) -> dict:
    r = requests.post(f"{OLLAMA_BASE_URL}{path}", json={**payload, "stream": False},
                      timeout=REQUEST_TIMEOUT)
    if not r.ok:  # Ollama はエラー理由を本文の "error" に入れて返す
        raise RuntimeError(f"Ollama {r.status_code}: {r.text[:300]}")
    return r.json()


def chat(engine: Engine, system: str, user: str, *, temperature: float = 0.1,
         max_tokens: int = 2048) -> str:
    """チャット形式で1回呼び出す（LM Studio / Ollama / Claude / OpenAI / Gemini）。"""
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
        record_anthropic(getattr(msg, "usage", None))  # トークン数の累計（「トークン数管理」タブ）
        return clean_output("".join(b.text for b in msg.content if b.type == "text"))

    if engine.provider in ("openai", "gemini"):
        return _cloud_chat(engine, system, user, max_tokens)

    if engine.provider == "ollama":
        res = _ollama("/api/chat", {
            "model": engine.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "think": False,  # 思考モデル（qwen3 など）の推論を止める（翻訳品質はほぼ同じで大幅に速い）
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": OLLAMA_NUM_CTX},
        })
        return clean_output(res["message"]["content"])

    res = _openai_client().chat.completions.create(
        model=engine.model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return clean_output(res.choices[0].message.content)


def _cloud_chat(engine: Engine, system: str, user: str, max_tokens: int) -> str:
    """OpenAI / Gemini を Chat Completions 形式で呼ぶ。

    最近のモデルは推論（思考）にも出力トークンを使うため、上限は CLOUD_MAX_TOKENS 以上にする。
    temperature を受け付けないモデルがあるため、環境変数で指定したときだけ送る。
    """
    prefix = engine.provider.upper()
    limit = max(max_tokens, CLOUD_MAX_TOKENS)
    kwargs: dict = {}
    if engine.provider == "openai":
        kwargs["max_completion_tokens"] = limit   # 新しいモデルは max_tokens ではなくこちら
    else:
        kwargs["max_tokens"] = limit
    if os.getenv(f"{prefix}_TEMPERATURE"):
        kwargs["temperature"] = float(os.environ[f"{prefix}_TEMPERATURE"])
    if os.getenv(f"{prefix}_REASONING_EFFORT"):
        kwargs["reasoning_effort"] = os.environ[f"{prefix}_REASONING_EFFORT"]
    res = _cloud_openai_client(engine.provider).chat.completions.create(
        model=engine.model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **kwargs,
    )
    # トークン数の累計（「トークン数管理」タブ）。出力が空でもトークンは消費されているので先に記録する
    record_openai_compatible(engine.provider, getattr(res, "usage", None))
    choice = res.choices[0]
    text = clean_output(choice.message.content or "")
    if not text:
        hint = "。CLOUD_MAX_TOKENS を増やしてください" if choice.finish_reason == "length" else ""
        raise RuntimeError(f"{PROVIDER_LABEL[engine.provider]} の出力が空でした"
                           f"（finish_reason={choice.finish_reason}{hint}）")
    return text


def complete(engine: Engine, prompt: str, *, stop: list[str] | None = None,
             max_tokens: int = 2048) -> str:
    """テキスト補完形式で1回呼び出す（plamo-2-translate / translategemma 用。LM Studio / Ollama）。"""
    if engine.provider == "ollama":
        # raw: モデルのチャットテンプレートを通さず、プロンプトをそのまま渡す
        res = _ollama("/api/generate", {
            "model": engine.model, "prompt": prompt, "raw": True,
            "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": OLLAMA_NUM_CTX,
                        **({"stop": stop} if stop else {})},
        })
        return clean_output(res["response"])

    res = _openai_client().completions.create(
        model=engine.model, prompt=prompt, stop=stop, max_tokens=max_tokens, temperature=0,
    )
    return clean_output(res.choices[0].text)
