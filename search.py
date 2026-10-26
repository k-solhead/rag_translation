"""過去翻訳のハイブリッド検索（pgvector + pg_trgm）。
 
検索の考え方:
    1. 3 系統で候補を集める
         - 完全一致（正規化後の md5 一致）
         - トライグラム類似（表記が近い文。「ほぼ同じ文」の検出に強い）
         - ベクトル類似（言い換え・語順違いなど意味が近い文）
    2. 候補ごとに両方の類似度を計算し、重み付きで合成スコアを出す
    3. 上位の候補について、翻訳メモリ風の一致率（単語単位）と
       「過去の英文 → 今回の英文」の差分を付けて返す
 
使い方:
    from common import Embedder, connect
    from search import hybrid_search
 
    embedder = Embedder()           # モデル読み込みは重いので、アプリ起動時に1回だけ
    with connect() as conn:
        for hit in hybrid_search(conn, embedder, "Press the power button to start."):
            print(hit["score"], hit["match_rate"], hit["diff_marked"], hit["target_ja"])
"""
from __future__ import annotations
 
import difflib
import re
from dataclasses import asdict, dataclass
 
from common import Embedder, clean_text, normalize_for_match
 
_TOKEN = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]", re.UNICODE)
 
_SQL = """
WITH
vec AS (            -- ベクトル類似の候補
    SELECT id FROM segments
     WHERE is_active
     ORDER BY embedding <=> %(qv)s
     LIMIT %(cand_k)s
),
trg AS (            -- トライグラム類似の候補（%% 演算子で similarity_threshold 以上を GIN で絞り込み）
    SELECT id FROM segments
     WHERE is_active AND source_norm %% %(qn)s
     ORDER BY similarity(source_norm, %(qn)s) DESC
     LIMIT %(cand_k)s
),
exact AS (          -- 完全一致
    SELECT id FROM segments
     WHERE is_active AND source_hash = md5(%(qn)s)
     LIMIT %(cand_k)s
),
cand AS (
    SELECT id FROM vec UNION SELECT id FROM trg UNION SELECT id FROM exact
)
SELECT s.id, d.doc_key, d.title, s.page_no, s.seq_no,
       s.source_en, s.target_ja,
       1 - (s.embedding <=> %(qv)s)          AS vec_sim,
       similarity(s.source_norm, %(qn)s)     AS trgm_sim,
       (s.source_hash = md5(%(qn)s))         AS exact
  FROM cand
  JOIN segments  s ON s.id = cand.id
  JOIN documents d ON d.id = s.document_id
"""
 
 
@dataclass
class Hit:
    segment_id: int
    doc_key: str
    title: str | None
    page_no: int | None
    seq_no: int | None
    source_en: str          # 過去の英文
    target_ja: str          # 過去の和訳
    score: float            # 合成スコア（0〜1、完全一致は 1.0）
    vec_sim: float          # コサイン類似度
    trgm_sim: float         # トライグラム類似度
    exact: bool             # 完全一致（正規化後）か
    match_rate: int         # 単語単位の一致率（%）。翻訳メモリの「◯%マッチ」相当
    diff_marked: str        # 差分表示: [-過去のみ-] {+今回のみ+}
    diff_ops: list[dict]    # 差分の詳細（画面でハイライトする場合に使用）
 
 
def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text)
 
 
def word_diff(old: str, new: str) -> tuple[int, str, list[dict]]:
    """過去の英文(old) → 今回の英文(new) の単語単位差分と一致率を返す。
 
    一致率は大文字小文字を無視して計算し、差分表示には原文の表記を使う。
    """
    old_t, new_t = _tokens(clean_text(old)), _tokens(clean_text(new))
    sm = difflib.SequenceMatcher(a=[t.lower() for t in old_t],
                                 b=[t.lower() for t in new_t], autojunk=False)
    rate = round(sm.ratio() * 100)
 
    marked: list[str] = []
    ops: list[dict] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        a, b = " ".join(old_t[i1:i2]), " ".join(new_t[j1:j2])
        if tag == "equal":
            marked.append(b)
        else:
            if a:
                marked.append(f"[-{a}-]")
            if b:
                marked.append(f"{{+{b}+}}")
            ops.append({"op": tag, "old": a, "new": b})
    return rate, " ".join(marked), ops
 
 
def hybrid_search(
    conn,
    embedder: Embedder,
    query_en: str,
    *,
    top_k: int = 5,
    cand_k: int = 50,
    w_trgm: float = 0.6,
    w_vec: float = 0.4,
    min_trgm: float = 0.3,
    min_score: float = 0.0,
) -> list[dict]:
    """英文 1 文に対して過去翻訳をハイブリッド検索する。
 
    Args:
        conn:       common.connect() で取得した接続
        embedder:   common.Embedder（投入時と同じモデル）
        query_en:   検索する英文
        top_k:      返す件数
        cand_k:     各系統で集める候補数
        w_trgm:     トライグラム類似度の重み（表記の近さ）
        w_vec:      ベクトル類似度の重み（意味の近さ）
        min_trgm:   トライグラム候補の下限類似度（pg_trgm.similarity_threshold）
        min_score:  合成スコアの下限（これ未満は返さない）
 
    Returns:
        Hit を dict にしたリスト（score の降順）
    """
    qn = normalize_for_match(query_en)
    if not qn:
        return []
    qv = embedder.encode([query_en])[0]
 
    with conn.transaction(), conn.cursor() as cur:
        # この検索のトランザクション内だけ有効な設定
        cur.execute("SELECT set_config('pg_trgm.similarity_threshold', %s, true)", (str(min_trgm),))
        cur.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(cand_k * 2, 40)),))
        cur.execute(_SQL, {"qv": qv, "qn": qn, "cand_k": cand_k})
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
 
    hits: list[Hit] = []
    for r in rows:
        vec_sim = float(r["vec_sim"] or 0.0)
        trgm_sim = float(r["trgm_sim"] or 0.0)
        score = 1.0 if r["exact"] else w_trgm * trgm_sim + w_vec * vec_sim
        if score < min_score:
            continue
        rate, marked, ops = word_diff(r["source_en"], query_en)
        hits.append(Hit(
            segment_id=r["id"], doc_key=r["doc_key"], title=r["title"],
            page_no=r["page_no"], seq_no=r["seq_no"],
            source_en=r["source_en"], target_ja=r["target_ja"],
            score=round(score, 4), vec_sim=round(vec_sim, 4), trgm_sim=round(trgm_sim, 4),
            exact=bool(r["exact"]), match_rate=100 if r["exact"] else rate,
            diff_marked=marked, diff_ops=ops,
        ))
 
    # 合成スコア → 一致率 の順で並べる。同じ英文・同じ訳の重複は1件にまとめる
    hits.sort(key=lambda h: (h.score, h.match_rate), reverse=True)
    seen: set[tuple[str, str]] = set()
    result: list[dict] = []
    for h in hits:
        key = (normalize_for_match(h.source_en), h.target_ja)
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
                  f"[{h['doc_key']} p.{h['page_no']}]")
            print(f"   過去英文: {h['source_en']}")
            print(f"   過去和訳: {h['target_ja']}")
            print(f"   差分    : {h['diff_marked']}")
 