"""RAG を使った翻訳（前年の訳を踏襲）と、結果の Excel 出力。

流れ（1文ごと）
    1. 原文で過去訳を RAG 検索（search.hybrid_search）
    2. 一致率で処理を分ける
         exact     : 完全一致（100%）→ 前年の訳文をそのまま使う（LLM に送らない）
         edit      : 一致率が edit_threshold 以上 → 前年の訳文を差分の箇所だけ修正させる
         reference : 一致率が ref_threshold 以上 → 新しく訳させ、過去訳は用語・文体の参考にする
         new       : 該当なし → 用語集だけを使って新しく訳させる
    3. 文に含まれる用語だけを用語集から抽出してプロンプトに入れる
    4. 選んだ複数のエンジンで並列に翻訳
    5. 自動チェック（数値の不一致・用語集の訳語の未使用・訳し残し）
    6. Excel に書き出す
"""
from __future__ import annotations

import io
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from common import Lang, detect_lang
from glossary import Term, match_terms
from llm import STYLE_LABEL, Engine, chat, complete
from search import hybrid_search

LANG_NAME = {"en": "英語", "ja": "日本語"}
MODE_LABEL = {"exact": "前年訳流用", "edit": "修正", "reference": "参考", "new": "新規", "empty": "－"}

Progress = Callable[[float, str], None]


# =====================================================================
# 入力ファイルの読み込みと文分割
# =====================================================================
def extract_from_docx(data: bytes) -> list[str]:
    """Word の段落（表のセル内の段落を含む）を文書の順に取り出す。"""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = docx.Document(io.BytesIO(data))
    texts: list[str] = []

    def walk(parent):
        for child in parent.iter_inner_content():
            if isinstance(child, Paragraph):
                if child.text.strip():
                    texts.append(child.text)
            elif isinstance(child, Table):
                for row in child.rows:
                    seen = set()
                    for cell in row.cells:
                        if id(cell._tc) in seen:  # 結合セルの重複を除く
                            continue
                        seen.add(id(cell._tc))
                        walk(cell)
    walk(doc)
    return texts


def extract_from_xlsx(data: bytes) -> list[str]:
    """Excel の全シートのセルを、上から順に1セル＝1段落として取り出す。"""
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    texts: list[str] = []
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            for cell in row:
                text = re.sub(r"\s+", " ", str(cell)).strip() if cell is not None else ""
                if text:
                    texts.append(text)
    wb.close()
    return texts


def extract_from_text(text: str) -> list[str]:
    """貼り付けたテキスト。空行または改行で段落を区切る。"""
    return [line.strip() for line in re.split(r"\n", text or "") if line.strip()]


_JA_OPEN, _JA_CLOSE, _JA_TERM = "「『（(【〔［[“", "」』）)】〕］]”", "。！？"


def split_ja(text: str) -> list[str]:
    """日本語を句点で文に分ける（括弧の中の句点では分けない）。"""
    out, buf, depth = [], [], 0
    for ch in text:
        buf.append(ch)
        if ch in _JA_OPEN:
            depth += 1
        elif ch in _JA_CLOSE:
            depth = max(0, depth - 1)
        elif ch in _JA_TERM and depth == 0:
            out.append("".join(buf).strip())
            buf = []
    if "".join(buf).strip():
        out.append("".join(buf).strip())
    return [s for s in out if s]


def split_en(text: str) -> list[str]:
    try:
        import pysbd
        seg = pysbd.Segmenter(language="en", clean=False)
        return [s.strip() for s in seg.segment(text) if s.strip()]
    except ImportError:
        return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"“(])", text) if s.strip()]


@dataclass
class Unit:
    no: int          # 通し番号
    para_no: int     # 段落番号（元ファイルでの順番）
    text: str


def make_units(paragraphs: list[str], lang: Lang, split_sentences: bool = True) -> list[Unit]:
    units: list[Unit] = []
    for p_no, para in enumerate(paragraphs, start=1):
        para = re.sub(r"\s+", " ", para or "").strip()  # 表記（全角・半角）は原文のまま保つ
        if not para:
            continue
        parts = (split_ja(para) if lang == "ja" else split_en(para)) if split_sentences else [para]
        for part in parts or [para]:
            units.append(Unit(len(units) + 1, p_no, part))
    return units


def detect_document_lang(paragraphs: list[str]) -> Lang:
    """文書全体の言語（和文の段落が半数以上なら和文）。"""
    sample = [p for p in paragraphs if p.strip()][:200]
    ja = sum(detect_lang(p) == "ja" for p in sample)
    return "ja" if sample and ja * 2 >= len(sample) else "en"


# =====================================================================
# プロンプト
# =====================================================================
SYSTEM_PROMPT = {
    "en": """あなたは統合報告書・サステナビリティレポートなど企業の開示資料を専門とするプロの翻訳者です。英語の原文を、自然で正確な日本語に翻訳します。

# 翻訳の方針
- <previous_translation>（前年の訳文）がある場合は、その語彙・文体（です・ます調／である調）・表記（数字、単位、記号、カタカナ語、送り仮名）を最優先で踏襲してください。
- <changes>（前年の原文からの変更点）に対応する箇所だけを修正し、それ以外の部分は前年の訳文の表現をそのまま残してください。
- <references>（過去の類似文とその訳文）がある場合は、用語と文体の参考にとどめ、内容は <source> に忠実に訳してください。
- <glossary>（必須の訳語）は必ず使ってください。
- <glossary_reference>（参考の訳語）は、文脈に合う場合に使ってください。前年の訳文がある場合は、前年の訳文の表現を優先してください。
- 数値・単位・年度・固有名詞・記号は原文どおり正確に反映してください。
- 出力は訳文のみとし、説明・注記・原文・引用符・タグは付けないでください。""",
    "ja": """You are a professional translator specializing in corporate disclosure documents such as integrated reports and sustainability reports. You translate Japanese source text into natural, accurate English suitable for corporate disclosure.

# Translation policy
- If <previous_translation> (last year's translation) is given, follow its vocabulary, style, and conventions (capitalization, number and unit formatting, spelling such as US/UK English, punctuation) as the top priority.
- Revise only the parts that correspond to <changes> (differences from last year's source) and keep the rest of last year's translation as it is.
- If <references> (similar past sentences and their translations) are given, use them only as a reference for terminology and style; translate the content faithfully from <source>.
- Always use the translations given in <glossary> (required terms).
- Use the translations in <glossary_reference> (reference terms) when they fit the context; if last year's translation is given, prefer its wording.
- Reflect numbers, units, fiscal years, proper nouns, and symbols exactly as in the source.
- Output only the translation, with no explanations, notes, source text, quotation marks, or tags.""",
}

_INSTRUCTION = {
    "en": "<source> を日本語に翻訳してください。訳文のみを出力してください。",
    "ja": "Translate <source> into English. Output only the translation.",
}

PLAMO_OP = "<|plamo:op|>"
PLAMO_EDIT_REPEAT = 3  # 修正モードで前年の対訳を並べる回数


def format_changes(diff_ops: list[dict], lang: Lang) -> str:
    """search.py の diff_ops（前年の原文 → 今年の原文）を、LLM 向けの変更点の箇条書きにする。"""
    lines = []
    for op in diff_ops:
        old, new = op["old"].strip(), op["new"].strip()
        if lang == "en":
            if old and new:
                lines.append(f"- 「{old}」→「{new}」に変更")
            elif new:
                lines.append(f"- 「{new}」を追加")
            elif old:
                lines.append(f"- 「{old}」を削除")
        else:
            if old and new:
                lines.append(f'- changed "{old}" to "{new}"')
            elif new:
                lines.append(f'- added "{new}"')
            elif old:
                lines.append(f'- removed "{old}"')
    return "\n".join(lines)


def format_glossary(terms: list[Term], lang: Lang) -> str:
    return "\n".join(f"{t.source(lang)} → {t.target(lang)}" for t in terms)


def build_rag_prompt(text: str, lang: Lang, mode: str, hits: list[dict],
                     terms: list[Term]) -> tuple[str, str]:
    """RAG 用の (system, user) プロンプト。"""
    parts: list[str] = []
    required = [t for t in terms if t.required]
    reference = [t for t in terms if not t.required]
    if required:
        parts.append(f"<glossary>\n{format_glossary(required, lang)}\n</glossary>")
    if reference:
        parts.append(f"<glossary_reference>\n{format_glossary(reference, lang)}\n</glossary_reference>")
    if mode == "edit" and hits:
        top = hits[0]
        parts.append(f"<previous_source>{top['source_text']}</previous_source>")
        parts.append(f"<previous_translation>{top['target_text']}</previous_translation>")
        changes = format_changes(top["diff_ops"], lang)
        if changes:
            parts.append(f"<changes>\n{changes}\n</changes>")
    elif mode == "reference" and hits:
        refs = "\n".join(
            f"<example>\n<source>{h['source_text']}</source>\n<translation>{h['target_text']}</translation>\n</example>"
            for h in hits)
        parts.append(f"<references>\n{refs}\n</references>")
    parts.append(f"<source>{text}</source>")
    parts.append(_INSTRUCTION[lang])
    return SYSTEM_PROMPT[lang], "\n".join(parts)


def build_plain_prompt(text: str, lang: Lang, terms: list[Term]) -> tuple[str, str]:
    """従来の translate と同じ短いプロンプト（ファインチューニング済みモデル用）。"""
    src, tgt = ("English", "Japanese") if lang == "en" else ("Japanese", "English")
    system = f"You are a helpful assistant that translates {src} to {tgt}."
    required = [t for t in terms if t.required]  # 従来プロンプトには必須の用語だけを入れる
    if required:
        system += f"\n\nglossary:\n{format_glossary(required, lang)}"
    return system, f"Translate the following {src} text to {tgt}: '{text}'"


def _one_line(text: str) -> str:
    return re.sub(r"\s*\n\s*", " ", text)


def build_plamo_prompt(text: str, lang: Lang, mode: str, hits: list[dict]) -> str:
    """plamo-2-translate 用。

    指示文を書けないモデルなので、過去の対訳を few-shot として並べて踏襲させる。
    修正モードは前年の対訳を繰り返すと踏襲が強まる（1回より3回のほうが前年訳に近い）。
    """
    src, tgt = ("English", "Japanese") if lang == "en" else ("Japanese", "English")
    shots = []
    if mode == "edit":
        shots = hits[:1] * PLAMO_EDIT_REPEAT
    elif mode == "reference":
        shots = hits[::-1]  # 最も似た文を今年の文の直前に置く
    examples = "".join(
        f"{PLAMO_OP}input lang={src}\n{_one_line(h['source_text'])}\n"
        f"{PLAMO_OP}output lang={tgt}\n{_one_line(h['target_text'])}\n"
        for h in shots)
    return (f"{PLAMO_OP}dataset\ntranslation\n{examples}"
            f"{PLAMO_OP}input lang={src}\n{_one_line(text)}\n{PLAMO_OP}output lang={tgt}\n")


def _gemma_turn(user: str, model: str | None = None) -> str:
    turn = f"<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"
    return turn if model is None else f"{turn}{model}<end_of_turn>\n"


def build_translategemma_prompt(text: str, lang: Lang, mode: str, hits: list[dict]) -> str:
    """translategemma 用。

    修正モードはポストエディットの指示文にする（公式には非サポートだが前年訳を最もよく踏襲する）。
    変更点の一覧は渡さなくても結果が変わらないので、前年と今年の原文だけを渡す。
    """
    (src, sc), (tgt, tc) = ((("English", "en"), ("Japanese", "ja")) if lang == "en"
                            else (("Japanese", "ja"), ("English", "en")))
    if mode == "edit" and hits:
        top = hits[0]
        return _gemma_turn(
            f"You are a professional {src} ({sc}) to {tgt} ({tc}) translator and post-editor. "
            f"Update the previous {tgt} translation so that it matches the new {src} text. "
            f"Change only what differs and keep all other wording exactly as it is. "
            f"Produce only the updated {tgt} translation, without any additional explanations or commentary.\n\n"
            f"Previous {src} text:\n{top['source_text']}\n\n"
            f"Previous {tgt} translation:\n{top['target_text']}\n\n"
            f"New {src} text:\n{text}"
        )
    intro = (
        f"You are a professional {src} ({sc}) to {tgt} ({tc}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {src} text "
        f"while adhering to {tgt} grammar, vocabulary, and cultural sensitivities. "
        f"Produce only the {tgt} translation, without any additional explanations or commentary. "
        f"Please translate the following {src} text into {tgt}:\n\n\n"
    )
    examples = ""
    if mode == "reference":
        # 最も似た文を今年の文の直前に置く
        examples = "".join(_gemma_turn(intro + h["source_text"], h["target_text"]) for h in hits[::-1])
    return examples + _gemma_turn(intro + text)


def run_engine(engine: Engine, text: str, lang: Lang, mode: str, hits: list[dict],
               terms: list[Term]) -> str:
    """1文を1つのエンジンで翻訳する。"""
    if engine.style == "plamo":
        return complete(engine, build_plamo_prompt(text, lang, mode, hits), stop=[PLAMO_OP])
    if engine.style == "translategemma":
        return complete(engine, build_translategemma_prompt(text, lang, mode, hits),
                        stop=["<end_of_turn>"])
    if engine.style == "plain":
        return chat(engine, *build_plain_prompt(text, lang, terms))
    return chat(engine, *build_rag_prompt(text, lang, mode, hits, terms))


# =====================================================================
# 自動チェック
# =====================================================================
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers(text: str) -> set[str]:
    t = unicodedata.normalize("NFKC", text)
    out = set()
    for raw in _NUM.findall(t):
        integer, _, frac = raw.replace(",", "").partition(".")
        out.add((integer.lstrip("0") or "0") + ("." + frac.rstrip("0") if frac.rstrip("0") else ""))
    return out


def check_translation(source: str, output: str, lang: Lang, terms: list[Term]) -> list[str]:
    """訳文の簡易チェック。問題点の短い説明のリストを返す。"""
    if not output:
        return ["訳文なし"]
    issues = []
    missing = _numbers(source) - _numbers(output)
    if missing:
        issues.append("数値差異（" + ", ".join(sorted(missing)[:3]) + "）")
    norm_out = unicodedata.normalize("NFKC", output).lower()
    unused = [t.target(lang) for t in terms if t.required  # 参考の用語はチェックしない
              and unicodedata.normalize("NFKC", t.target(lang)).lower() not in norm_out]
    if unused:
        issues.append("用語未使用（" + ", ".join(unused[:3]) + "）")
    if (lang == "en") == (detect_lang(output) == "en"):  # 訳文が原文と同じ言語のまま
        issues.append("訳し残しの可能性")
    return issues


# =====================================================================
# 一括翻訳
# =====================================================================
@dataclass
class TranslationSettings:
    lang: Lang                              # 原文の言語
    engines: list[Engine]
    doc_ids: list[int] | None = None        # RAG 検索の対象文書（None = 全文書）
    glossary: list[Term] = field(default_factory=list)
    edit_threshold: int = 75                # これ以上は「修正」モード
    ref_threshold: int = 40                 # これ以上は「参考」モード（未満は「新規」）
    ref_count: int = 3                      # 参考モードで渡す過去訳の件数
    reuse_exact: bool = True                # 完全一致は LLM に送らず前年訳を使う


@dataclass
class RowResult:
    unit: Unit
    mode: str
    hits: list[dict]
    terms: list[Term]
    outputs: dict[str, str] = field(default_factory=dict)   # engine.key → 訳文
    errors: dict[str, str] = field(default_factory=dict)    # engine.key → エラー
    checks: dict[str, list[str]] = field(default_factory=dict)

    @property
    def top(self) -> dict | None:
        return self.hits[0] if self.hits else None


def plan_units(conn, embedder, units: list[Unit], st: TranslationSettings,
               progress: Progress) -> list[RowResult]:
    """各文について RAG 検索し、処理モードと用語を決める。"""
    rows: list[RowResult] = []
    for i, u in enumerate(units):
        hits = hybrid_search(conn, embedder, u.text, lang=st.lang, doc_ids=st.doc_ids,
                             top_k=max(st.ref_count, 1))
        top = hits[0] if hits else None
        rate = top["match_rate"] if top else 0
        if top and (top["exact"] or rate >= 100):
            mode = "exact"
        elif rate >= st.edit_threshold:
            mode = "edit"
        elif rate >= st.ref_threshold:
            mode = "reference"
        else:
            mode = "new"
        refs = [h for h in hits if h["match_rate"] >= st.ref_threshold][:st.ref_count]
        rows.append(RowResult(u, mode, refs if mode == "reference" else hits[:1] if top else [],
                              match_terms(st.glossary, u.text, st.lang)))
        progress((i + 1) / len(units), f"過去訳を検索中… {i + 1}/{len(units)}")
    return rows


def translate_rows(rows: list[RowResult], st: TranslationSettings, progress: Progress) -> None:
    """全エンジン × 全文を並列に翻訳する（エンジンごとに同時実行数を制限）。"""
    jobs = []
    for r in rows:
        for e in st.engines:
            if r.mode == "exact" and st.reuse_exact and r.top:
                r.outputs[e.key] = r.top["target_text"]
            else:
                jobs.append((r, e))
    total = len(jobs)
    done = 0
    if total == 0:
        progress(1.0, "翻訳対象はありません（すべて前年訳を流用）")
    pools = {e.key: ThreadPoolExecutor(max_workers=max(1, e.concurrency)) for e in st.engines}
    try:
        futures = {
            pools[e.key].submit(run_engine, e, r.unit.text, st.lang,
                                "edit" if r.mode == "exact" else r.mode, r.hits, r.terms): (r, e)
            for r, e in jobs
        }
        for fut in as_completed(futures):
            r, e = futures[fut]
            try:
                r.outputs[e.key] = fut.result()
            except Exception as exc:  # 1件の失敗で全体を止めない
                r.outputs[e.key] = ""
                r.errors[e.key] = f"{type(exc).__name__}: {str(exc)[:200]}"
            done += 1
            progress(done / total, f"翻訳中… {done}/{total}（{e.label}）")
    finally:
        for p in pools.values():
            p.shutdown(wait=False, cancel_futures=True)

    for r in rows:
        for e in st.engines:
            if e.key in r.errors:
                r.checks[e.key] = ["エラー: " + r.errors[e.key]]
            elif not (r.mode == "exact" and st.reuse_exact):
                r.checks[e.key] = check_translation(r.unit.text, r.outputs.get(e.key, ""), st.lang, r.terms)


def run_translation(conn, embedder, units: list[Unit], st: TranslationSettings,
                    progress: Progress | None = None) -> list[RowResult]:
    report = progress or (lambda *_: None)
    rows = plan_units(conn, embedder, units, st, lambda f, m: report(0.3 * f, m))
    translate_rows(rows, st, lambda f, m: report(0.3 + 0.7 * f, m))
    return rows


# =====================================================================
# Excel 出力
# =====================================================================
_DIFF = re.compile(r"\[-(.*?)-\]|\{\+(.*?)\+\}", re.S)
MODE_FILL = {"exact": "E2F0D9", "edit": "FFF4CC", "reference": "DDEBF7", "new": None}


def _diff_rich_text(marked: str):
    from openpyxl.cell.rich_text import CellRichText, TextBlock
    from openpyxl.cell.text import InlineFont

    parts, pos = [], 0
    for m in _DIFF.finditer(marked):
        if m.start() > pos:
            parts.append(marked[pos:m.start()])
        if m.group(1) is not None:
            parts.append(TextBlock(InlineFont(color="C00000", strike=True), m.group(1)))
        else:
            parts.append(TextBlock(InlineFont(color="00792B", b=True), m.group(2)))
        pos = m.end()
    if pos < len(marked):
        parts.append(marked[pos:])
    return CellRichText(*parts) if parts else ""


def results_to_excel(rows: list[RowResult], st: TranslationSettings, meta: dict[str, str]) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    src, tgt = LANG_NAME[st.lang], LANG_NAME["ja" if st.lang == "en" else "en"]
    wb = Workbook()
    ws = wb.active
    ws.title = "翻訳結果"
    headers = (["No.", "段落", f"原文（{src}）", "一致率", "処理", f"前年の原文（{src}）",
                "差分（前年→今年）", f"前年の訳文（{tgt}）", "参照文書"]
               + [e.label for e in st.engines] + ["用語集（該当）", "チェック"])
    ws.append(headers)
    widths = [6, 6, 50, 7, 10, 45, 45, 45, 20] + [50] * len(st.engines) + [30, 40]

    for r in rows:
        top = r.top if r.mode != "new" else None
        ref_docs = "、".join(dict.fromkeys(
            f"{h['title'] or h['doc_key']}" + (f" p.{h['page_source']}" if h["page_source"] else "")
            for h in r.hits))
        checks = "\n".join(
            f"{e.model}/{STYLE_LABEL[e.style]}: {'・'.join(r.checks[e.key])}"
            for e in st.engines if r.checks.get(e.key))
        ws.append([
            r.unit.no, r.unit.para_no, r.unit.text,
            r.top["match_rate"] if r.top else None, MODE_LABEL[r.mode],
            top["source_text"] if top else "", None,
            top["target_text"] if top else "", ref_docs,
            *[r.outputs.get(e.key, "") for e in st.engines],
            "\n".join(f"{t.source(st.lang)} → {t.target(st.lang)}" + ("" if t.required else "（参考）")
                      for t in r.terms), checks,
        ])
        if top and top["diff_ops"]:
            ws.cell(ws.max_row, 7).value = _diff_rich_text(top["diff_marked"])

    header_fill = PatternFill("solid", fgColor="173F35")
    thin = Border(bottom=Side(style="thin", color="D9E2DE"))
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = w
    check_col = len(headers)
    for row_cells, r in zip(ws.iter_rows(min_row=2), rows):
        fill = MODE_FILL.get(r.mode)
        for cell in row_cells:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = thin
        if fill:
            for cell in row_cells[:5]:
                cell.fill = PatternFill("solid", fgColor=fill)
        if row_cells[check_col - 1].value:
            row_cells[check_col - 1].font = Font(color="C00000")
    ws.freeze_panes = "D2"
    ws.auto_filter.ref = ws.dimensions

    info = wb.create_sheet("実行条件")
    info.append(["項目", "内容"])
    counts = {m: sum(r.mode == m for r in rows) for m in MODE_LABEL}
    for k, v in [
        ("実行日時", datetime.now().strftime("%Y-%m-%d %H:%M")),
        *meta.items(),
        ("翻訳の向き", f"{src} → {tgt}"),
        ("エンジン", "\n".join(e.label for e in st.engines)),
        ("処理の判定", f"一致率{st.edit_threshold}%以上＝修正、{st.ref_threshold}%以上＝参考、それ未満＝新規"
                       + ("、完全一致＝前年訳流用（LLMに送らない）" if st.reuse_exact else "")),
        ("処理の内訳", "、".join(f"{MODE_LABEL[m]} {n}文" for m, n in counts.items() if n)),
        ("色分け", "緑＝前年訳流用、黄＝修正、青＝参考、色なし＝新規。チェック列の赤字は要確認"),
    ]:
        info.append([k, v])
    info.column_dimensions["A"].width = 16
    info.column_dimensions["B"].width = 100
    for row_cells in info.iter_rows():
        for cell in row_cells:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        row_cells[0].font = Font(bold=True)

    gl = wb.create_sheet("使用した用語")
    gl.append([src, tgt, "区分", "顧客"])
    used = {(t.source(st.lang), t.target(st.lang), t.kind, t.client) for r in rows for t in r.terms}
    for s, t, k, c in sorted(used):
        gl.append([s, t, k, c or "（共通）"])
    for col, w in zip("ABCD", (40, 40, 8, 24)):
        gl.column_dimensions[col].width = w

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
