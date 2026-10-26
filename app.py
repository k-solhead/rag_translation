import html
import re
from io import BytesIO

import streamlit as st

from common import Embedder, connect
from ingest import ingest_rows, read_excel_rows
from search import hybrid_search

_DIFF = re.compile(r"\[-(.*?)-\]|\{\+(.*?)\+\}")


@st.cache_resource(show_spinner="埋め込みモデルを読み込んでいます...")
def get_embedder() -> Embedder:
    return Embedder()


def diff_html(marked: str) -> str:
    """search.py の diff_marked（[-過去のみ-] {+今回のみ+}）を色付き HTML にする。"""
    out: list[str] = []
    pos = 0
    for m in _DIFF.finditer(marked):
        out.append(html.escape(marked[pos:m.start()]))
        if m.group(1) is not None:
            out.append(
                '<span style="background:#ffd6d6;color:#b00020;text-decoration:line-through">'
                f"{html.escape(m.group(1))}</span>"
            )
        else:
            out.append(
                '<span style="background:#d6f5d6;color:#0a6b0a;font-weight:bold">'
                f"{html.escape(m.group(2))}</span>"
            )
        pos = m.end()
    out.append(html.escape(marked[pos:]))
    return "".join(out)


def rate_color(rate: int) -> str:
    if rate >= 100:
        return "#0a6b0a"
    if rate >= 75:
        return "#b36b00"
    return "#666666"


st.title("RAG Translation App")
search_tab, ingest_tab = st.tabs(["過去翻訳を検索", "データ取り込み"])

with search_tab:
    query = st.text_area("英文を貼り付けてください", height=150, key="query_en")
    col1, col2 = st.columns(2)
    top_k = col1.slider("表示件数", 1, 20, 5)
    min_rate = col2.slider("最低一致率（%）", 0, 100, 0)

    if st.button("検索", type="primary"):
        if not query.strip():
            st.warning("英文を入力してください。")
        else:
            try:
                with st.spinner("検索しています..."):
                    with connect() as conn:
                        hits = hybrid_search(conn, get_embedder(), query, top_k=top_k)
            except Exception as exc:
                st.error(f"検索に失敗しました: {exc}")
            else:
                hits = [h for h in hits if h["match_rate"] >= min_rate]
                hits.sort(key=lambda h: (h["match_rate"], h["score"]), reverse=True)
                if not hits:
                    st.info("該当する過去翻訳が見つかりませんでした。")
                for i, h in enumerate(hits, 1):
                    page = f" p.{h['page_no']}" if h["page_no"] is not None else ""
                    label = html.escape(h["title"] or h["doc_key"])
                    with st.container(border=True):
                        st.markdown(
                            f'<span style="font-size:1.4em;font-weight:bold;'
                            f'color:{rate_color(h["match_rate"])}">{h["match_rate"]}%</span>'
                            f"　#{i}　<small>{label}{page}"
                            f"　(score {h['score']:.3f} / trgm {h['trgm_sim']:.3f}"
                            f" / vec {h['vec_sim']:.3f})</small>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("**過去の英文**")
                        st.write(h["source_en"])
                        st.markdown("**過去の和訳**")
                        st.write(h["target_ja"])
                        st.markdown("**差分（過去の英文 → 今回の英文）**")
                        if not h["diff_ops"]:
                            st.caption("差分なし")
                        else:
                            st.markdown(
                                f'<div style="line-height:1.8">{diff_html(h["diff_marked"])}</div>',
                                unsafe_allow_html=True,
                            )

with ingest_tab:
    uploaded_file = st.file_uploader(
        "Excelファイル（.xlsx）をアップロード",
        type=["xlsx"],
        key="en_xlsx",
    )

    st.caption(
        "先頭シートに doc_key、source_en、target_ja の列が必要です。"
        "title と page_no は任意です。"
    )

    if uploaded_file and st.button("取り込む"):
        try:
            rows = read_excel_rows(BytesIO(uploaded_file.getvalue()))
        except ValueError as exc:
            st.error(str(exc))
        else:
            if not rows:
                st.error("投入対象がありません。必須項目が入力された行を確認してください。")
            else:
                with st.spinner("データを取り込んでいます..."):
                    inserted = ingest_rows(rows)
                st.success(f"取り込みが完了しました（新規 {inserted:,} 行）")