"""対訳（英文・和文の組）から用語集の候補を抽出する。

抽出方法
    1. 短い対訳 : 見出し・表の項目など、短い対訳をそのまま用語の候補にする（LLM 不要）
    2. LLM      : 文中の固有名詞・取り組み名・専門用語などを LLM に抜き出させる

どちらの候補も、対訳全体で検証する。
    文中の出現 : 英文にその用語が出てくる対訳の数
    一貫性     : そのうち、和文に候補の訳語が使われている割合
LLM の候補で、和文に訳語が一度も出てこないもの（LLM の思い違い）は除く。

既存の用語集と照合し「新規／登録済み／訳語が違う」を付ける。既存の用語も一貫性を出すので、
必須（毎回この訳語）と参考（文脈で訳し分ける）の見直しにも使える。
"""
from __future__ import annotations

import io
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from glossary import Term, _clean, strip_wrapping
from llm import Engine, chat

Progress = Callable[[float, str], None]
_JA = re.compile(r"[぀-ヿ㐀-鿿]")
_LETTER = re.compile(r"[A-Za-z]")
_ACRONYM = re.compile(r"\b[A-Z]{2,}\d*\b|\d")
REQUIRED_TYPES = {"固有名詞", "取り組み名", "略語", "指標・目標"}


@dataclass
class Pair:
    en: str
    ja: str


@dataclass
class Candidate:
    en: str
    ja: str
    method: str                     # 短い対訳 / LLM / 短い対訳＋LLM / 既存
    count: int = 0                  # 短い対訳として出た回数、または LLM が挙げた回数
    in_sentence: int = 0            # 英文に用語が出てくる対訳の数
    matched: int = 0                # そのうち和文に訳語がある数
    type: str = ""                  # LLM が付けた種別
    status: str = "新規"            # 新規 / 登録済み / 訳語が違う
    existing_ja: str = ""
    variants: list[tuple[str, int]] = field(default_factory=list)  # 短い対訳での他の訳
    example_en: str = ""
    example_ja: str = ""
    required: bool = False
    adopt: bool = False

    @property
    def consistency(self) -> float | None:
        return self.matched / self.in_sentence if self.in_sentence else None


# =====================================================================
# 対訳の取得
# =====================================================================
def pairs_from_db(conn, doc_ids: list[int] | None) -> list[Pair]:
    sql = "SELECT en_text, ja_text FROM segments WHERE is_active"
    params: list = []
    if doc_ids is not None:
        sql += " AND document_id = ANY(%s)"
        params.append(doc_ids)
    return [Pair(en, ja) for en, ja in conn.execute(sql + " ORDER BY document_id, seq_no", params)]


def pairs_from_rows(rows: list[dict]) -> list[Pair]:
    """ingest.read_excel / read_csv の行（en_text / ja_text）から。"""
    return [Pair(r["en_text"], r["ja_text"]) for r in rows]


# =====================================================================
# 検証用の索引
# =====================================================================
def _norm_en(text: str) -> str:
    return _clean(text).lower()


def _norm_ja(text: str) -> str:
    return re.sub(r"\s+", "", _clean(text))


class PairIndex:
    """用語が英文に出てくる対訳を探し、和文に訳語が使われているかを数える。"""

    def __init__(self, pairs: list[Pair]):
        self.pairs = pairs
        self.en = [_norm_en(p.en) for p in pairs]
        self.ja = [_norm_ja(p.ja) for p in pairs]

    def stats(self, en_term: str, ja_term: str) -> tuple[int, int, int | None]:
        """(英文に出る数, そのうち和文に訳語がある数, 例文の番号)"""
        key, ja_key = _norm_en(en_term), _norm_ja(ja_term)
        if not key:
            return 0, 0, None
        pat = re.compile(r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])")
        n = m = 0
        example = None
        for i, en in enumerate(self.en):
            if key in en and pat.search(en):
                n += 1
                if ja_key and ja_key in self.ja[i]:
                    m += 1
                    if example is None or len(self.en[i]) < len(self.en[example]):
                        example = i  # 短い例文を優先
        return n, m, example


# =====================================================================
# 1. 短い対訳
# =====================================================================
def _clean_term(text: str) -> str:
    text = strip_wrapping(_clean(text))
    return text.rstrip(":：").strip()


def extract_short_pairs(pairs: list[Pair], *, max_words: int = 4, max_ja: int = 20,
                        min_count: int = 2) -> list[Candidate]:
    groups: dict[str, Counter] = defaultdict(Counter)
    forms: dict[str, Counter] = defaultdict(Counter)
    for p in pairs:
        en, ja = _clean_term(p.en), _clean_term(p.ja)
        words = en.split()
        if not (1 <= len(words) <= max_words and len(ja) <= max_ja):
            continue
        if not _LETTER.search(en) or not _JA.search(ja) or _JA.search(en):
            continue
        if re.search(r"[.!?]$", en) and len(words) > 1:  # 文
            continue
        key = en.lower()
        groups[key][ja] += 1
        forms[key][en] += 1
    out = []
    for key, ja_counts in groups.items():
        total = sum(ja_counts.values())
        if total < min_count:
            continue
        (ja, _), *others = ja_counts.most_common()
        out.append(Candidate(forms[key].most_common(1)[0][0], ja, "短い対訳", count=total,
                             variants=[(j, c) for j, c in others]))
    return out


# =====================================================================
# 2. LLM
# =====================================================================
EXTRACT_SYSTEM = """あなたは、統合報告書・サステナビリティレポートなど企業の開示資料の翻訳用語集を作る専門家です。
与えられた英日の対訳から、用語集に登録すべき用語の対を抽出してください。

# 抽出する用語
- 固有名詞（会社名・ブランド名・製品名・団体名・制度名・認証名など）
- 取り組み・プログラム・プロジェクトの名称
- 目標・指標の名称
- 業界・専門分野の用語（サステナビリティ、環境、人権、ガバナンスなど）
- 略語（正式名称が分かる場合は正式名称も別の対として）

# 抽出しない用語
- 一般的な語（例: company, employees, Japan, increase など）
- 数値だけのもの、文全体

# ルール
- en は英文中の表記、ja は和文中の表記をそのまま使ってください（和文に無い訳語を作らないこと）。
- type は「固有名詞」「取り組み名」「指標・目標」「専門用語」「略語」のいずれか。
- 出力は次の JSON だけにしてください。該当がなければ {"terms": []}。
{"terms": [{"en": "...", "ja": "...", "type": "..."}]}"""


def _parse_terms(text: str) -> list[dict]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return []
    terms = data.get("terms", []) if isinstance(data, dict) else []
    return [t for t in terms if isinstance(t, dict) and t.get("en") and t.get("ja")]


def extract_with_llm(pairs: list[Pair], engine: Engine, *, min_words: int = 5, batch_size: int = 30,
                     max_pairs: int | None = None, progress: Progress | None = None
                     ) -> tuple[list[Candidate], list[str]]:
    """文の対訳（min_words 語以上）を LLM に渡して用語を抜き出す。(候補, エラー) を返す。"""
    report = progress or (lambda *_: None)
    seen, targets = set(), []
    for p in pairs:
        key = (_clean(p.en), _clean(p.ja))
        if len(key[0].split()) >= min_words and key not in seen:
            seen.add(key)
            targets.append(Pair(*key))
    if max_pairs:
        targets = targets[:max_pairs]
    batches = [targets[i:i + batch_size] for i in range(0, len(targets), batch_size)]
    if not batches:
        return [], []

    def run(batch: list[Pair]) -> list[dict]:
        user = "\n".join(f"{i}. EN: {p.en}\n   JA: {p.ja}" for i, p in enumerate(batch, 1))
        return _parse_terms(chat(engine, EXTRACT_SYSTEM, user, temperature=0, max_tokens=4096))

    counts: Counter = Counter()
    types: dict[tuple[str, str], Counter] = defaultdict(Counter)
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, engine.concurrency)) as pool:
        futures = [pool.submit(run, b) for b in batches]
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                for t in fut.result():
                    en, ja = _clean_term(str(t["en"])), _clean_term(str(t["ja"]))
                    if en and ja:
                        counts[(en, ja)] += 1
                        types[(en, ja)][str(t.get("type", ""))] += 1
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {str(exc)[:150]}")
            report(k / len(batches), f"LLM で用語を抽出中… {k}/{len(batches)}")
    return [Candidate(en, ja, "LLM", count=n, type=types[(en, ja)].most_common(1)[0][0])
            for (en, ja), n in counts.items()], errors


# =====================================================================
# 候補のまとめ・検証・既存用語集との照合
# =====================================================================
def _default_required(c: Candidate) -> bool:
    if c.method == "既存":
        return c.consistency is None or c.consistency >= 0.85
    if c.type in REQUIRED_TYPES or _ACRONYM.search(c.en):
        return True
    capitalized = sum(w[:1].isupper() for w in c.en.split())
    return capitalized >= 2 and (c.consistency is None or c.consistency >= 0.9)


def build_candidates(pairs: list[Pair], short: list[Candidate], llm: list[Candidate],
                     existing: list[Term], *, include_existing: bool = True,
                     min_consistency: float = 0.8, progress: Progress | None = None) -> list[Candidate]:
    report = progress or (lambda *_: None)
    index = PairIndex(pairs)

    merged: dict[tuple[str, str], Candidate] = {}
    for c in short + llm:
        key = (c.en.lower(), _norm_ja(c.ja))
        if key in merged:
            m = merged[key]
            m.count += c.count
            m.type = m.type or c.type
            if c.method not in m.method:
                m.method = "短い対訳＋LLM"
        else:
            merged[key] = c

    existing_map: dict[str, list[Term]] = defaultdict(list)
    for t in existing:
        existing_map[t.en.lower()].append(t)
    if include_existing:
        for t in existing:
            key = (t.en.lower(), _norm_ja(t.ja))
            if key not in merged:
                merged[key] = Candidate(t.en, t.ja, "既存", required=t.required)

    out: list[Candidate] = []
    items = list(merged.values())
    for i, c in enumerate(items, 1):
        c.in_sentence, c.matched, ex = index.stats(c.en, c.ja)
        if ex is not None:
            c.example_en, c.example_ja = index.pairs[ex].en, index.pairs[ex].ja
        if i % 50 == 0 or i == len(items):
            report(i / len(items), f"対訳全体で検証中… {i}/{len(items)}")
        if c.method == "既存" and c.in_sentence == 0:
            continue  # 今回の対訳に出てこない既存の用語は表示しない
        if c.method == "LLM" and c.matched == 0:
            continue  # 和文に訳語が見当たらない（LLM の思い違い）
        same = existing_map.get(c.en.lower(), [])
        if any(_norm_ja(t.ja) == _norm_ja(c.ja) for t in same):
            c.status = "登録済み"
        elif same:
            c.status = "訳語が違う"
            c.existing_ja = "、".join(dict.fromkeys(t.ja for t in same))
        c.required = _default_required(c)
        cons = c.consistency
        c.adopt = (c.status == "新規" and (cons is None or cons >= min_consistency)
                   and (c.count >= 2 or c.matched >= 2))
        out.append(c)

    order = {"新規": 0, "訳語が違う": 1, "登録済み": 2}
    out.sort(key=lambda c: (order[c.status], -(c.count + c.matched), c.en.lower()))
    return out


# =====================================================================
# 出力
# =====================================================================
COLUMNS = ["採用", "en", "ja", "区分", "抽出方法", "種別", "既存との関係", "既存の訳語",
           "出現回数", "文中の出現", "一貫性(%)", "他の訳例", "例文（英）", "例文（日）"]


def candidates_to_records(cands: list[Candidate]) -> list[dict]:
    return [{
        "採用": c.adopt, "en": c.en, "ja": c.ja, "区分": "必須" if c.required else "参考",
        "抽出方法": c.method, "種別": c.type, "既存との関係": c.status, "既存の訳語": c.existing_ja,
        "出現回数": c.count, "文中の出現": c.in_sentence,
        "一貫性(%)": None if c.consistency is None else round(c.consistency * 100),
        "他の訳例": "、".join(f"{j}({n})" for j, n in c.variants[:3]),
        "例文（英）": c.example_en, "例文（日）": c.example_ja,
    } for c in cands]


def candidates_to_excel(records: list[dict]) -> bytes:
    """候補一覧の Excel。「採用」列に ○ を付けた行だけ、用語集タブの「ファイルから取り込む」で取り込める。"""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = Workbook()
    ws = wb.active
    ws.title = "用語候補"
    ws.append(COLUMNS)
    for r in records:
        ws.append([("○" if v else "") if k == "採用" else v for k, v in ((k, r.get(k)) for k in COLUMNS)])
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="173F35")
    widths = [6, 34, 30, 7, 12, 10, 12, 24, 9, 9, 9, 24, 50, 40]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = w
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=cell.column > 11)
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions
    notes = wb.create_sheet("説明")
    for line in [
        "「採用」列に ○ を付けた行だけが取り込まれます（用語集タブ →「ファイルから取り込む」）。",
        "区分：必須＝毎回この訳語を使う（翻訳時の自動チェック対象）／参考＝文脈に合えば使う訳語の候補。",
        "文中の出現：英文に用語が出てくる対訳の数。一貫性：そのうち和文に訳語が使われている割合。",
        "一貫性が低い用語は、文脈で訳し分けている可能性があります（参考にするか、登録を見送ってください）。",
        "既存との関係：新規＝用語集に無い／登録済み＝同じ訳語で登録済み／訳語が違う＝別の訳語で登録済み。",
    ]:
        notes.append([line])
    notes.column_dimensions["A"].width = 110
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
