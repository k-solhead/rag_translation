"""過去翻訳のハイブリッド検索（pgvector + pg_trgm）― 英→和・和→英の両方向対応。

検索の考え方:
    1. 検索文の言語（英文 / 和文）に合わせて、segments の en_* か ja_* の列を使う
    2. 3 系統で候補を集める
         - 完全一致（正規化後の md5 一致）
         - トライグラム類似（表記が近い文。「ほぼ同じ文」の検出に強い）
         - ベクトル類似（言い換え・語順違いなど意味が近い文）
    3. 候補ごとに両方の類似度を計算し、重み付きで合成スコアを出す
    4. 上位の候補について、一致率と「過去の文 → 今回の文」の差分を付けて返す
         - 英文: 単語単位の一致率・差分
         - 和文: 文字単位の一致率、形態素（単語）単位の差分

文書の絞り込み（doc_ids）を指定した場合、ベクトル検索は索引（HNSW）を使わず、
対象文書の全行と比較する。索引を使うと絞り込み後の候補が大きく欠けることがあるため。

使い方:
    from common import Embedder, connect
    from search import hybrid_search

    embedder = Embedder()           # モデル読み込みは重いので、アプリ起動時に1回だけ
    with connect() as conn:
        hits = hybrid_search(conn, embedder, "電源ボタンを押して本体の電源を入れます。")
        for h in hits:
            print(h["match_rate"], h["diff_marked"], h["target_text"])
"""
from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass

from common import Embedder, Lang, detect_lang, normalize_for_match, tokenize

# 検索言語ごとの列（SQL に埋め込むのはこの固定値だけ。入力値は埋め込まない）
_COLS: dict[str, dict[str, str]] = {
    lang: {
        "text": f"{lang}_text", "norm": f"{lang}_norm", "hash": f"{lang}_hash",
        "emb": f"{lang}_embedding", "page": f"page_{lang}",
    }
    for lang in ("en", "ja")
}

_SQL = """
WITH
vec AS (            -- ベクトル類似の候補
    SELECT id FROM segments
     WHERE is_active AND {emb} IS NOT NULL {doc_filter}
     ORDER BY {vec_order}
     LIMIT %(cand_k)s
),
trg AS (            -- トライグラム類似の候補（%% 演算子で similarity_threshold 以上を GIN で絞り込み）
    SELECT id FROM segments
     WHERE is_active AND {norm} %% %(qn)s {doc_filter}
     ORDER BY similarity({norm}, %(qn)s) DESC
     LIMIT %(cand_k)s
),
exact AS (          -- 完全一致
    SELECT id FROM segments
     WHERE is_active AND {hash} = md5(%(qn)s) {doc_filter}
     LIMIT %(cand_k)s
),
cand AS (
    SELECT id FROM vec UNION SELECT id FROM trg UNION SELECT id FROM exact
)
SELECT s.id, d.doc_key, d.title, d.client, d.tags, s.seq_no,
       s.{src_text} AS source_text, s.{tgt_text} AS target_text,
       s.{src_page} AS page_source, s.{tgt_page} AS page_target,
       1 - (s.{emb} <=> %(qv)s)          AS vec_sim,
       similarity(s.{norm}, %(qn)s)      AS trgm_sim,
       (s.{hash} = md5(%(qn)s))          AS exact
  FROM cand
  JOIN segments  s ON s.id = cand.id
  JOIN documents d ON d.id = s.document_id
"""


@dataclass
class Hit:
    segment_id: int
    doc_key: str
    title: str | None
    client: str | None
    tags: list[str]
    seq_no: int | None
    source_lang: str        # 検索した言語（"en" / "ja"）
    target_lang: str        # 訳文の言語
    source_text: str        # 過去の原文（検索文と同じ言語）
    target_text: str        # 過去の訳文
    page_source: int | None
    page_target: int | None
    score: float            # 合成スコア（0〜1、完全一致は 1.0）
    vec_sim: float          # コサイン類似度
    trgm_sim: float         # トライグラム類似度
    exact: bool             # 完全一致（正規化後）か
    match_rate: int         # 一致率（%）。翻訳メモリの「◯%マッチ」相当
    diff_marked: str        # 差分表示: [-過去のみ-]{+今回のみ+}
    diff_ops: list[dict]    # 差分の詳細（画面でハイライトする場合に使用）


def match_rate(old: str, new: str, lang: Lang) -> int:
    """一致率（%）。英文は単語単位、和文は文字単位（空白・全角半角の違いは無視）。"""
    if lang == "ja":
        a, b = normalize_for_match(old, "ja"), normalize_for_match(new, "ja")
    else:
        a = [t.lower() for t in tokenize(old, "en")]
        b = [t.lower() for t in tokenize(new, "en")]
    return round(difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio() * 100)


def text_diff(old: str, new: str, lang: Lang) -> tuple[str, list[dict]]:
    """過去の文(old) → 今回の文(new) の差分。英文は単語、和文は形態素単位。

    大文字小文字の違いは無視して比べ、表示には原文の表記を使う。
    """
    sep = " " if lang == "en" else ""
    old_t, new_t = tokenize(old, lang), tokenize(new, lang)
    sm = difflib.SequenceMatcher(a=[t.lower() for t in old_t],
                                 b=[t.lower() for t in new_t], autojunk=False)
    marked: list[str] = []
    ops: list[dict] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        a, b = sep.join(old_t[i1:i2]), sep.join(new_t[j1:j2])
        if tag == "equal":
            marked.append(b)
            continue
        part = (f"[-{a}-]" if a.strip() else "") + (f"{{+{b}+}}" if b.strip() else "")
        if part:
            marked.append(part)
            ops.append({"op": tag, "old": a, "new": b})
    return sep.join(marked), ops


def hybrid_search(
    conn,
    embedder: Embedder,
    query: str,
    *,
    lang: Lang | None = None,
    doc_ids: list[int] | None = None,
    top_k: int = 5,
    cand_k: int = 50,
    w_trgm: float = 0.6,
    w_vec: float = 0.4,
    min_trgm: float = 0.3,
    min_score: float = 0.0,
) -> list[dict]:
    """英文または和文 1 文に対して過去翻訳をハイブリッド検索する。

    Args:
        conn:       common.connect() で取得した接続
        embedder:   common.Embedder（投入時と同じモデル）
        query:      検索する文（英文 / 和文）
        lang:       検索文の言語。None なら自動判定（かな・漢字を含めば和文）
        doc_ids:    検索対象の文書 id。None なら全文書（[] なら該当なし）
        top_k:      返す件数
        cand_k:     各系統で集める候補数
        w_trgm:     トライグラム類似度の重み（表記の近さ）
        w_vec:      ベクトル類似度の重み（意味の近さ）
        min_trgm:   トライグラム候補の下限類似度（pg_trgm.similarity_threshold）
        min_score:  合成スコアの下限（これ未満は返さない）

    Returns:
        Hit を dict にしたリスト（score の降順）
    """
    lang = lang or detect_lang(query)
    if lang not in _COLS:
        raise ValueError(f"lang は 'en' か 'ja' を指定してください: {lang!r}")
    target = "ja" if lang == "en" else "en"
    qn = normalize_for_match(query, lang)
    if not qn or doc_ids == []:
        return []
    qv = embedder.encode([query])[0]

    src, tgt = _COLS[lang], _COLS[target]
    filtered = doc_ids is not None
    sql = _SQL.format(
        emb=src["emb"], norm=src["norm"], hash=src["hash"],
        src_text=src["text"], tgt_text=tgt["text"], src_page=src["page"], tgt_page=tgt["page"],
        doc_filter="AND document_id = ANY(%(doc_ids)s)" if filtered else "",
        # 絞り込み時は「+ 0」で索引を使わせず、対象文書の全行と比較する（取りこぼし防止）
        vec_order=f"({src['emb']} <=> %(qv)s) + 0" if filtered else f"{src['emb']} <=> %(qv)s",
    )
    params = {"qv": qv, "qn": qn, "cand_k": cand_k, "doc_ids": list(doc_ids or [])}

    with conn.transaction(), conn.cursor() as cur:
        # この検索のトランザクション内だけ有効な設定
        cur.execute("SELECT set_config('pg_trgm.similarity_threshold', %s, true)", (str(min_trgm),))
        cur.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(cand_k * 2, 40)),))
        cur.execute(sql, params)
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    hits: list[Hit] = []
    for r in rows:
        vec_sim = float(r["vec_sim"] or 0.0)
        trgm_sim = float(r["trgm_sim"] or 0.0)
        score = 1.0 if r["exact"] else w_trgm * trgm_sim + w_vec * vec_sim
        if score < min_score:
            continue
        marked, ops = text_diff(r["source_text"], query, lang)
        hits.append(Hit(
            segment_id=r["id"], doc_key=r["doc_key"], title=r["title"], client=r["client"],
            tags=list(r["tags"] or []), seq_no=r["seq_no"],
            source_lang=lang, target_lang=target,
            source_text=r["source_text"], target_text=r["target_text"],
            page_source=r["page_source"], page_target=r["page_target"],
            score=round(score, 4), vec_sim=round(vec_sim, 4), trgm_sim=round(trgm_sim, 4),
            exact=bool(r["exact"]),
            match_rate=100 if r["exact"] else match_rate(r["source_text"], query, lang),
            diff_marked=marked, diff_ops=ops,
        ))

    # 合成スコア → 一致率 の順で並べる。同じ原文・同じ訳文の重複は1件にまとめる
    hits.sort(key=lambda h: (h.score, h.match_rate), reverse=True)
    seen: set[tuple[str, str]] = set()
    result: list[dict] = []
    for h in hits:
        key = (normalize_for_match(h.source_text, lang), h.target_text)
        if key in seen:
            continue
        seen.add(key)
        result.append(asdict(h))
        if len(result) >= top_k:
            break
    return result


if __name__ == "__main__":
    import sys

    from common import connect

    q = " ".join(sys.argv[1:]) or "Press the power button to turn on the device."
    emb = Embedder()
    with connect() as c:
        for i, h in enumerate(hybrid_search(c, emb, q), 1):
            print(f"#{i} score={h['score']:.3f} match={h['match_rate']}% "
                  f"(trgm={h['trgm_sim']:.3f} vec={h['vec_sim']:.3f}) "
                  f"[{h['doc_key']} p.{h['page_source']}]")
            print(f"   過去の原文: {h['source_text']}")
            print(f"   過去の訳文: {h['target_text']}")
            print(f"   差分      : {h['diff_marked']}")
