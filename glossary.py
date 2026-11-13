"""用語集（共通・顧客・媒体ごと）の DB 管理と、文ごとの用語の抽出。

glossary_terms テーブル
    client : 顧客名（空文字 '' = 全顧客共通の用語集）
    medium : 媒体名（空文字 '' = その顧客の会社全体の用語集。例：統合報告書、サステナビリティレポート）
    en     : 英語の用語
    ja     : 日本語の訳語
    note     : 備考（任意）
    required : True＝必須（必ずこの訳語を使う）／False＝参考（文脈に合えば使う）

用語集は「共通 → 会社 → 媒体」の 3 段階。翻訳時は「共通」＋「選んだ顧客（会社全体）」
＋「選んだ媒体」の用語集を重ねて使い、各文に含まれる用語だけを LLM に渡す。
同じ英語の用語が複数にある場合は、媒体 ＞ 会社 ＞ 共通 の順に優先する
（媒体の用語集には、会社の用語集と訳語が違う用語だけを登録すればよい）。
必須の用語は「必ず使う訳語」としてプロンプトに入れ、自動チェック（用語未使用）の対象にする。
参考の用語は「訳語の候補」としてだけ示し、チェックの対象にしない。
"""
from __future__ import annotations

import csv
import io
import re
import unicodedata
from dataclasses import dataclass

COMMON = ""  # 共通用語集を表す client の値
COMMON_LABEL = "（共通）"
COMPANY_WIDE = ""  # 会社全体（媒体を指定しない）を表す medium の値

GLOSSARY_DDL = """
CREATE TABLE IF NOT EXISTS glossary_terms (
    id          bigserial    PRIMARY KEY,
    client      text         NOT NULL DEFAULT '',   -- 顧客名（'' = 共通）
    medium      text         NOT NULL DEFAULT '',   -- 媒体名（'' = 会社全体）
    en          text         NOT NULL,
    ja          text         NOT NULL,
    note        text,
    required    boolean      NOT NULL DEFAULT true, -- true=必須 / false=参考
    created_at  timestamptz  NOT NULL DEFAULT now(),
    updated_at  timestamptz  NOT NULL DEFAULT now(),
    CONSTRAINT glossary_terms_client_medium_en_ja_key UNIQUE (client, medium, en, ja)
);
CREATE INDEX IF NOT EXISTS idx_glossary_terms_client_medium ON glossary_terms (client, medium);
DROP TRIGGER IF EXISTS trg_glossary_terms_updated_at ON glossary_terms;
CREATE TRIGGER trg_glossary_terms_updated_at
    BEFORE UPDATE ON glossary_terms
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
"""

COLUMN_ALIASES = {
    "en": ("en", "英語", "英文", "English", "english", "EN"),
    "ja": ("ja", "日本語", "和文", "Japanese", "japanese", "JA"),
    "note": ("note", "備考", "メモ", "Note"),
    "required": ("required", "区分", "必須", "種別"),
    "adopt": ("採用",),
}
_FALSE_WORDS = {"参考", "false", "0", "no", "×", "x", "-", "いいえ", "任意"}
_TRUE_WORDS = {"必須", "true", "1", "yes", "○", "◯", "o", "はい", "✓"}
_WRAP_PAIRS = [("「", "」"), ("『", "』"), ("“", "”"), ('"', '"'), ("‘", "’"), ("'", "'")]
_JA_CHAR = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


@dataclass(frozen=True)
class Term:
    en: str
    ja: str
    client: str = COMMON
    note: str | None = None
    required: bool = True
    medium: str = COMPANY_WIDE

    @property
    def kind(self) -> str:
        return "必須" if self.required else "参考"

    def source(self, lang: str) -> str:
        return self.en if lang == "en" else self.ja

    def target(self, lang: str) -> str:
        """lang（原文の言語）から見た訳語。"""
        return self.ja if lang == "en" else self.en


def _clean(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("º", "°")  # 序数記号 º は NFKC で o になるため、度記号として扱う
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def strip_wrapping(text: str) -> str:
    """全体を囲む括弧・引用符（「…」『…』“…” など）を外す。途中の括弧はそのまま。"""
    text = text.strip()
    for left, right in _WRAP_PAIRS:
        if len(text) <= 2 or not (text.startswith(left) and text.endswith(right)):
            continue
        # 囲みが1組だけのときに限る（「A」と「B」のような文字列は外さない）
        once = text.count(left) == 2 if left == right else text.count(left) == text.count(right) == 1
        if once:
            return text[1:-1].strip()
    return text


def parse_required(value, default: bool = True) -> bool:
    if value is None or str(value).strip() == "":
        return default
    if isinstance(value, bool):
        return value
    word = str(value).strip().lower()
    if word in _FALSE_WORDS:
        return False
    if word in _TRUE_WORDS:
        return True
    return default


def client_label(client: str) -> str:
    return client or COMMON_LABEL


def glossary_label(client: str, medium: str = COMPANY_WIDE) -> str:
    """画面表示用の用語集の名前（例：（共通）／アサヒ／アサヒ・統合報告書）。"""
    if client and medium:
        return f"{client}・{medium}"
    return client_label(client)


def glossary_scope_label(client: str | None, medium: str | None = None) -> str:
    """翻訳で重ねて使う用語集の説明（例：アサヒ・統合報告書（＋会社全体＋共通））。"""
    if not client:
        return "共通のみ"
    if medium:
        return f"{client}・{medium}（＋会社全体＋共通）"
    return f"{client}（＋共通）"


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
    # 後から追加した列（必須／参考、媒体）
    conn.execute("ALTER TABLE glossary_terms ADD COLUMN IF NOT EXISTS required boolean NOT NULL DEFAULT true")
    conn.execute("ALTER TABLE glossary_terms ADD COLUMN IF NOT EXISTS medium text NOT NULL DEFAULT ''")
    # 一意制約を (client, en, ja) → (client, medium, en, ja) に付け替える（既存のデータは medium = '' になる）
    names = {r[0] for r in conn.execute(
        "SELECT conname FROM pg_constraint WHERE conrelid = 'glossary_terms'::regclass AND contype = 'u'")}
    if "glossary_terms_client_medium_en_ja_key" not in names:
        if "glossary_terms_client_en_ja_key" in names:
            conn.execute("ALTER TABLE glossary_terms DROP CONSTRAINT glossary_terms_client_en_ja_key")
        conn.execute("ALTER TABLE glossary_terms ADD CONSTRAINT glossary_terms_client_medium_en_ja_key "
                     "UNIQUE (client, medium, en, ja)")
    conn.execute("DROP INDEX IF EXISTS idx_glossary_terms_client")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_glossary_terms_client_medium ON glossary_terms (client, medium)")
    conn.commit()


def list_glossaries(conn) -> list[tuple[str, str]]:
    """用語が登録されている (顧客名, 媒体名) の組（共通 ('', '') を含む）。"""
    return [tuple(r) for r in conn.execute(
        "SELECT DISTINCT client, medium FROM glossary_terms ORDER BY client, medium")]


def list_glossary_clients(conn) -> list[str]:
    """用語が登録されている顧客名（共通 '' を含む。媒体だけに用語がある顧客も含む）。"""
    return sorted({c for c, _ in list_glossaries(conn)})


def list_media(conn, client: str) -> list[str]:
    """その顧客で用語が登録されている媒体名（会社全体 '' は含まない）。"""
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT medium FROM glossary_terms WHERE client = %s AND medium <> '' ORDER BY medium",
        (client,))]


def count_terms(conn) -> dict[tuple[str, str], int]:
    """(顧客名, 媒体名) ごとの用語数。"""
    return {(c, m): n for c, m, n in conn.execute(
        "SELECT client, medium, count(*) FROM glossary_terms GROUP BY client, medium")}


def get_terms(conn, client: str, medium: str = COMPANY_WIDE) -> list[Term]:
    rows = conn.execute(
        "SELECT en, ja, client, note, required, medium FROM glossary_terms"
        " WHERE client = %s AND medium = %s ORDER BY en, ja",
        (client, medium if client else COMPANY_WIDE))
    return [Term(*r) for r in rows]


def save_terms(conn, client: str, terms: list[Term], *, replace: bool, medium: str = COMPANY_WIDE) -> int:
    """用語を登録する。replace=True ならその用語集（顧客・媒体）を丸ごと置き換える。"""
    medium = _clean(medium) if client else COMPANY_WIDE  # 共通の用語集に媒体はない
    terms = [t for t in terms if _clean(t.en) and _clean(t.ja)]
    with conn.transaction():
        if replace:
            conn.execute("DELETE FROM glossary_terms WHERE client = %s AND medium = %s", (client, medium))
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO glossary_terms (client, medium, en, ja, note, required) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (client, medium, en, ja) DO UPDATE
                    SET note = COALESCE(EXCLUDED.note, glossary_terms.note), required = EXCLUDED.required
                """,
                [(client, medium, _clean(t.en), _clean(t.ja), (t.note or "").strip() or None, bool(t.required))
                 for t in terms],
            )
    return len(terms)


def delete_terms(conn, client: str, medium: str = COMPANY_WIDE) -> None:
    """その用語集（顧客・媒体）の用語をすべて削除する。媒体を指定しなければ会社全体の用語集だけ。"""
    conn.execute("DELETE FROM glossary_terms WHERE client = %s AND medium = %s",
                 (client, medium if client else COMPANY_WIDE))
    conn.commit()


def load_glossary(conn, client: str | None, medium: str | None = None) -> list[Term]:
    """翻訳に使う用語集（共通＋会社全体＋媒体）。同じ英語の用語は 媒体 ＞ 会社 ＞ 共通 の順に優先する。"""
    layers = [get_terms(conn, COMMON)]
    if client:
        layers.append(get_terms(conn, client))
        if medium:
            layers.append(get_terms(conn, client, medium))
    terms: list[Term] = []
    seen: set[str] = set()
    for layer in reversed(layers):  # 優先度の高い順
        terms += [t for t in layer if t.en.lower() not in seen]
        seen |= {t.en.lower() for t in layer}
    return terms


# ---------------------------------------------------------------------
# ファイル入出力
# ---------------------------------------------------------------------
def _pick(record: dict, key: str):
    for name in COLUMN_ALIASES[key]:
        if record.get(name) not in (None, ""):
            return record[name]
    return None


def _looks_like_header(cells: list[str]) -> bool:
    names = set(cells)
    return bool(names & set(COLUMN_ALIASES["en"])) and bool(names & set(COLUMN_ALIASES["ja"]))


def read_glossary_file(data: bytes, filename: str, *, strip_quotes: bool = True) -> list[Term]:
    """CSV（UTF-8 / BOM付き / Shift_JIS）または Excel（先頭シート）から用語を読む。

    列名あり : en / ja（英語・日本語、英文・和文も可）、note（備考）、required（区分＝必須／参考）、
               採用（○の行だけ取り込む。用語抽出の候補一覧をそのまま取り込む場合）
    列名なし : 1列目＝英語、2列目＝日本語、3列目＝備考（日本語が1列目なら自動で入れ替え）
    strip_quotes : 全体を囲む「」『』“” などを外す（文中で使うときの括弧は翻訳時に文脈で付ける）
    """
    if filename.lower().endswith((".xlsx", ".xlsm")):
        from openpyxl import load_workbook
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        table = [["" if v is None else v for v in r] for r in wb.worksheets[0].iter_rows(values_only=True)]
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
        table = list(csv.reader(io.StringIO(text)))
    table = [r for r in table if any(str(v).strip() for v in r)]
    if not table:
        return []

    first = [str(v).strip() for v in table[0]]
    if _looks_like_header(first):
        records = [dict(zip(first, r)) for r in table[1:]]
    else:
        if len(first) < 2:
            raise ValueError("用語集は「英語」「日本語」の2列以上が必要です")
        sample = table[:200]
        ja_first = sum(bool(_JA_CHAR.search(str(r[0]))) for r in sample if r)
        ja_second = sum(bool(_JA_CHAR.search(str(r[1]))) for r in sample if len(r) > 1)
        en_col, ja_col = (1, 0) if ja_first > ja_second else (0, 1)
        records = [{"en": r[en_col] if len(r) > en_col else "", "ja": r[ja_col] if len(r) > ja_col else "",
                    "note": r[2] if len(r) > 2 else ""} for r in table]

    has_adopt = any(_pick(r, "adopt") is not None for r in records)
    terms, seen = [], set()
    for r in records:
        if has_adopt and not parse_required(_pick(r, "adopt"), default=False):
            continue
        en, ja = _clean(_pick(r, "en")), _clean(_pick(r, "ja"))
        if strip_quotes:
            en, ja = strip_wrapping(en), strip_wrapping(ja)
        if en and ja and (en, ja) not in seen:
            seen.add((en, ja))
            terms.append(Term(en, ja, note=_clean(_pick(r, "note")) or None,
                              required=parse_required(_pick(r, "required"))))
    return terms


def glossary_to_excel(terms: list[Term]) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    wb = Workbook()
    ws = wb.active
    ws.title = "用語集"
    ws.append(["en", "ja", "区分", "note"])
    for t in terms:
        ws.append([t.en, t.ja, t.kind, t.note or ""])
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="173F35")
    for col, width in zip("ABCD", (40, 40, 8, 40)):
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
