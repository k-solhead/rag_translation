"""用語集（顧客ごと）の DB 管理と、文ごとの用語の抽出。

glossary_terms テーブル
    client : 顧客名（空文字 '' = 全顧客共通の用語集）
    en     : 英語の用語
    ja     : 日本語の訳語
    note   : 備考（任意）

翻訳時は「共通」＋「選んだ顧客」の用語集を使い、各文に含まれる用語だけを LLM に渡す。
同じ用語が両方にある場合は、顧客の用語集を優先する。
"""
from __future__ import annotations

import csv
import io
import re
import unicodedata
from dataclasses import dataclass

COMMON = ""  # 共通用語集を表す client の値
COMMON_LABEL = "（共通）"

GLOSSARY_DDL = """
CREATE TABLE IF NOT EXISTS glossary_terms (
    id          bigserial    PRIMARY KEY,
    client      text         NOT NULL DEFAULT '',   -- 顧客名（'' = 共通）
    en          text         NOT NULL,
    ja          text         NOT NULL,
    note        text,
    created_at  timestamptz  NOT NULL DEFAULT now(),
    updated_at  timestamptz  NOT NULL DEFAULT now(),
    UNIQUE (client, en, ja)
);
CREATE INDEX IF NOT EXISTS idx_glossary_terms_client ON glossary_terms (client);
DROP TRIGGER IF EXISTS trg_glossary_terms_updated_at ON glossary_terms;
CREATE TRIGGER trg_glossary_terms_updated_at
    BEFORE UPDATE ON glossary_terms
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
"""

COLUMN_ALIASES = {
    "en": ("en", "英語", "英文", "English", "english", "EN"),
    "ja": ("ja", "日本語", "和文", "Japanese", "japanese", "JA"),
    "note": ("note", "備考", "メモ", "Note"),
}


@dataclass(frozen=True)
class Term:
    en: str
    ja: str
    client: str = COMMON
    note: str | None = None

    def source(self, lang: str) -> str:
        return self.en if lang == "en" else self.ja

    def target(self, lang: str) -> str:
        """lang（原文の言語）から見た訳語。"""
        return self.ja if lang == "en" else self.en


def _clean(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def client_label(client: str) -> str:
    return client or COMMON_LABEL


# ---------------------------------------------------------------------
# DB 操作
# ---------------------------------------------------------------------
def ensure_glossary_table(conn) -> None:
    """用語集テーブルが無ければ作る（既存の DB を作り直さずに機能を追加するため）。"""
    conn.execute("""
        CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
        BEGIN NEW.updated_at := now(); RETURN NEW; END;
        $$ LANGUAGE plpgsql
    """)
    exists = conn.execute("SELECT to_regclass('public.glossary_terms') IS NOT NULL").fetchone()[0]
    if not exists:
        conn.execute(GLOSSARY_DDL)
    conn.commit()


def list_glossary_clients(conn) -> list[str]:
    """用語が登録されている顧客名（共通 '' を含む）。"""
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT client FROM glossary_terms ORDER BY client")]


def count_terms(conn) -> dict[str, int]:
    return dict(conn.execute("SELECT client, count(*) FROM glossary_terms GROUP BY client"))


def get_terms(conn, client: str) -> list[Term]:
    rows = conn.execute(
        "SELECT en, ja, client, note FROM glossary_terms WHERE client = %s ORDER BY en, ja",
        (client,))
    return [Term(*r) for r in rows]


def save_terms(conn, client: str, terms: list[Term], *, replace: bool) -> int:
    """用語を登録する。replace=True ならその顧客の用語集を丸ごと置き換える。"""
    terms = [t for t in terms if _clean(t.en) and _clean(t.ja)]
    with conn.transaction():
        if replace:
            conn.execute("DELETE FROM glossary_terms WHERE client = %s", (client,))
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO glossary_terms (client, en, ja, note) VALUES (%s, %s, %s, %s)
                ON CONFLICT (client, en, ja) DO UPDATE SET note = EXCLUDED.note
                """,
                [(client, _clean(t.en), _clean(t.ja), (t.note or "").strip() or None) for t in terms],
            )
    return len(terms)


def delete_client_terms(conn, client: str) -> None:
    conn.execute("DELETE FROM glossary_terms WHERE client = %s", (client,))


def load_glossary(conn, client: str | None) -> list[Term]:
    """翻訳に使う用語集（共通＋顧客）。同じ原語は顧客の用語集を優先する。"""
    terms = get_terms(conn, COMMON)
    if client:
        own = get_terms(conn, client)
        own_en = {t.en.lower() for t in own}
        terms = own + [t for t in terms if t.en.lower() not in own_en]
    return terms


# ---------------------------------------------------------------------
# ファイル入出力
# ---------------------------------------------------------------------
def _pick(record: dict, key: str):
    for name in COLUMN_ALIASES[key]:
        if record.get(name) not in (None, ""):
            return record[name]
    return None


def read_glossary_file(data: bytes, filename: str) -> list[Term]:
    """CSV（UTF-8 / BOM付き / Shift_JIS）または Excel（先頭シート）から用語を読む。

    列名: en / ja（英語・日本語、英文・和文も可）、note（備考・メモも可・任意）
    """
    if filename.lower().endswith((".xlsx", ".xlsm")):
        from openpyxl import load_workbook
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        rows = wb.worksheets[0].iter_rows(values_only=True)
        headers = [str(h).strip() if h is not None else "" for h in (next(rows, None) or ())]
        records = [dict(zip(headers, r)) for r in rows]
        wb.close()
    else:
        for enc in ("utf-8-sig", "cp932"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("CSV の文字コードを判別できません（UTF-8 か Shift_JIS で保存してください）")
        reader = csv.DictReader(io.StringIO(text))
        headers = [h.strip() for h in (reader.fieldnames or [])]
        records = [{(k or "").strip(): v for k, v in r.items()} for r in reader]

    for key in ("en", "ja"):
        if not set(headers) & set(COLUMN_ALIASES[key]):
            raise ValueError(f"用語集に「{' / '.join(COLUMN_ALIASES[key][:3])}」の列が必要です")
    terms, seen = [], set()
    for r in records:
        en, ja = _clean(_pick(r, "en")), _clean(_pick(r, "ja"))
        if en and ja and (en, ja) not in seen:
            seen.add((en, ja))
            terms.append(Term(en, ja, note=_clean(_pick(r, "note")) or None))
    return terms


def glossary_to_excel(terms: list[Term]) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    wb = Workbook()
    ws = wb.active
    ws.title = "用語集"
    ws.append(["en", "ja", "note"])
    for t in terms:
        ws.append([t.en, t.ja, t.note or ""])
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="173F35")
    for col, width in zip("ABC", (40, 40, 40)):
        ws.column_dimensions[col].width = width
    ws.freeze_panes = "A2"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------
# 文ごとの用語抽出
# ---------------------------------------------------------------------
def match_terms(terms: list[Term], text: str, lang: str) -> list[Term]:
    """text（原文）に含まれる用語だけを返す。

    英文: 大文字小文字を区別せず、単語の途中では一致させない（"CO2" が "CO2e" の一部、などを除外）
    和文: 全角半角・空白の違いを無視した部分一致
    長い用語を優先し、長い用語の一部にすぎない短い用語は除く。
    """
    norm = _clean(text)
    if lang == "en":
        hay = norm.lower()
    else:
        hay = re.sub(r"\s+", "", norm)
    found: list[Term] = []
    spans: list[tuple[int, int]] = []
    for t in sorted(terms, key=lambda t: len(t.source(lang)), reverse=True):
        src = _clean(t.source(lang))
        if not src:
            continue
        if lang == "en":
            m = re.search(r"(?<![A-Za-z0-9])" + re.escape(src.lower()) + r"(?![A-Za-z0-9])", hay)
        else:
            m = re.search(re.escape(re.sub(r"\s+", "", src)), hay)
        if not m:
            continue
        if any(s <= m.start() and m.end() <= e for s, e in spans):
            continue  # より長い用語の一部
        spans.append((m.start(), m.end()))
        found.append(t)
    return found
