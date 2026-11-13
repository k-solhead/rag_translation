"""クラウド LLM（OpenAI / Claude / Gemini）のトークン使用量の累計。

データベース（token_usage テーブル）に保存するため、アプリやコンテナを再起動しても消えず、
アプリを使う全員の使用分が合算される。

    列                   内容
    input_tokens         入力トークン（キャッシュ以外。通常の入力単価で課金される分）
    output_tokens        出力トークン（推論・思考に使った分を含む）
    cache_read_tokens    キャッシュ入力（キャッシュからの読み込み。割引単価の分）
    cache_write_tokens   キャッシュ出力（キャッシュへの書き込み。割増単価の分）
    requests             呼び出し回数

4 つのトークン数にそれぞれの単価を掛けて足すと、おおよその料金になる。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from threading import Lock

from common import connect

PROVIDERS = {"openai": "OpenAI", "anthropic": "Claude", "gemini": "Gemini"}  # 画面の表示順
FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")

USAGE_DDL = """
CREATE TABLE IF NOT EXISTS token_usage (
    provider            text         PRIMARY KEY,           -- anthropic / openai / gemini
    input_tokens        bigint       NOT NULL DEFAULT 0,    -- 入力（キャッシュ以外）
    output_tokens       bigint       NOT NULL DEFAULT 0,    -- 出力（推論分を含む）
    cache_read_tokens   bigint       NOT NULL DEFAULT 0,    -- キャッシュ入力（読み込み）
    cache_write_tokens  bigint       NOT NULL DEFAULT 0,    -- キャッシュ出力（書き込み）
    requests            bigint       NOT NULL DEFAULT 0,    -- 呼び出し回数
    updated_at          timestamptz,                        -- 最後に加算した日時
    reset_at            timestamptz  NOT NULL DEFAULT now() -- 集計開始（最後にゼロに戻した）日時
)
"""

_log = logging.getLogger(__name__)
_table_ready = False
_table_lock = Lock()


def ensure_usage_table(conn) -> None:
    """累計テーブルが無ければ作る（既存の DB を作り直さずに機能を追加するため）。"""
    global _table_ready
    conn.execute(USAGE_DDL)
    conn.commit()
    _table_ready = True


def record_usage(provider: str, *, input_tokens: int = 0, output_tokens: int = 0,
                 cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> None:
    """1 回の呼び出し分を加算する。DB 側で足し込むため、並列で呼ばれても数は狂わない。

    記録に失敗しても例外は出さない（トークン数の記録のために翻訳を止めない）。
    """
    if provider not in PROVIDERS:
        return
    values = [max(0, int(v or 0)) for v in (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)]
    try:
        with connect() as conn:
            if not _table_ready:
                with _table_lock:
                    ensure_usage_table(conn)
            conn.execute(
                """
                INSERT INTO token_usage AS t (provider, input_tokens, output_tokens,
                                              cache_read_tokens, cache_write_tokens, requests, updated_at)
                VALUES (%s, %s, %s, %s, %s, 1, now())
                ON CONFLICT (provider) DO UPDATE SET
                    input_tokens       = t.input_tokens       + EXCLUDED.input_tokens,
                    output_tokens      = t.output_tokens      + EXCLUDED.output_tokens,
                    cache_read_tokens  = t.cache_read_tokens  + EXCLUDED.cache_read_tokens,
                    cache_write_tokens = t.cache_write_tokens + EXCLUDED.cache_write_tokens,
                    requests           = t.requests + 1,
                    updated_at         = now()
                """,
                (provider, *values),
            )
    except Exception as exc:  # noqa: BLE001
        _log.warning("トークン数の記録に失敗しました（%s）: %s", provider, exc)


def get_usage(conn) -> dict[str, dict]:
    """プロバイダーごとの累計。まだ記録のないプロバイダーは 0 で返す。"""
    rows = {p: {**{f: 0 for f in FIELDS}, "requests": 0, "updated_at": None, "reset_at": None}
            for p in PROVIDERS}
    cur = conn.execute(f"SELECT provider, {', '.join(FIELDS)}, requests, updated_at, reset_at FROM token_usage")
    cols = [c.name for c in cur.description]
    for r in cur.fetchall():
        d = dict(zip(cols, r))
        if d["provider"] in rows:
            rows[d.pop("provider")] = d
    return rows


def reset_usage(conn, provider: str) -> None:
    """そのプロバイダーの累計をすべてゼロに戻し、集計開始日時を今にする。"""
    conn.execute(
        """
        INSERT INTO token_usage (provider, reset_at) VALUES (%s, now())
        ON CONFLICT (provider) DO UPDATE SET
            input_tokens = 0, output_tokens = 0, cache_read_tokens = 0, cache_write_tokens = 0,
            requests = 0, updated_at = NULL, reset_at = now()
        """,
        (provider,),
    )
    conn.commit()


# ---------------------------------------------------------------------
# 応答の usage からの読み取り
# ---------------------------------------------------------------------
def _get(obj, *names) -> int:
    """obj の属性（または dict のキー）を順に探し、最初に見つかった数値を返す。"""
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def record_anthropic(usage) -> None:
    """Claude（Messages API）の usage。input_tokens はキャッシュ分を含まない。"""
    if usage is None:
        return
    record_usage(
        "anthropic",
        input_tokens=_get(usage, "input_tokens"),
        output_tokens=_get(usage, "output_tokens"),
        cache_read_tokens=_get(usage, "cache_read_input_tokens"),
        cache_write_tokens=_get(usage, "cache_creation_input_tokens"),
    )


def record_openai_compatible(provider: str, usage) -> None:
    """OpenAI / Gemini（Chat Completions 形式）の usage。

    prompt_tokens はキャッシュ分を含むので、キャッシュ分を引いて「キャッシュ以外の入力」にする。
    キャッシュへの書き込み数は、応答に項目があるときだけ記録する（無ければ 0）。
    """
    if usage is None:
        return
    details = (usage.get("prompt_tokens_details") if isinstance(usage, dict)
               else getattr(usage, "prompt_tokens_details", None)) or {}
    cache_read = _get(details, "cached_tokens")
    cache_write = _get(details, "cache_write_tokens", "cache_creation_tokens", "cache_creation_input_tokens")
    prompt = _get(usage, "prompt_tokens")
    record_usage(
        provider,
        input_tokens=max(0, prompt - cache_read - cache_write),
        output_tokens=_get(usage, "completion_tokens"),
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


JST = timezone(timedelta(hours=9), "JST")


def format_time(value: datetime | None) -> str:
    """日本時間で表示する（コンテナの時刻設定は UTC のことが多いため）。"""
    return value.astimezone(JST).strftime("%Y-%m-%d %H:%M") if value else "－"
