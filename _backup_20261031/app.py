"""RAG Translation App（Streamlit）― 過去翻訳の検索（英→和・和→英）、対訳表の登録、文書管理。"""
import html
import re
from io import BytesIO
from pathlib import Path

import pandas as pd
import psycopg
import streamlit as st

from common import LANG_LABEL, Embedder, connect, detect_lang
from documents import (
    DocInfo, delete_document, find_document_ids, get_document, list_clients, list_documents,
    list_tags, list_titles, update_document,
)
from ingest import Bitext, ingest_bitext, read_csv, read_excel
from search import hybrid_search

st.set_page_config(page_title="RAG Translation App", page_icon="🔎", layout="wide")

_DIFF = re.compile(r"\[-(.*?)-\]|\{\+(.*?)\+\}", re.S)
DIRECTIONS = {
    "自動判定": None,
    "英→和（英文で検索）": "en",
    "和→英（和文で検索）": "ja",
}


@st.cache_resource(show_spinner="埋め込みモデルを読み込んでいます...")
def get_embedder() -> Embedder:
    return Embedder()


def diff_html(marked: str) -> str:
    """search.py の diff_marked（[-過去のみ-]{+今回のみ+}）を色付き HTML にする。"""
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


def flash(message: str) -> None:
    """再描画（st.rerun）後に1回だけ表示するメッセージ。"""
    st.session_state["flash"] = message


def doc_info_inputs(key: str, info: DocInfo, clients: list[str], tags: list[str]) -> DocInfo:
    """案件名・ファイル名・顧客名・タグ・メモの入力欄。key ごとに独立した入力状態を持つ。"""
    c1, c2 = st.columns(2)
    title = c1.text_input("案件名", value=info.title or "", key=f"{key}_title",
                          placeholder="例）アサヒ_サスレポ2026_和文")
    doc_key = c2.text_input("ファイル名（必須・一意）", value=info.doc_key or "", key=f"{key}_doc_key",
                            placeholder="例）AGH_SR26J")
    client_options = sorted(set(clients) | ({info.client} if info.client else set()))
    client = c1.selectbox(
        "顧客名", client_options,
        index=client_options.index(info.client) if info.client else None,
        accept_new_options=True, key=f"{key}_client",
        placeholder="選択、または入力して Enter（例：アサヒグループホールディングス）",
    )
    tag_options = sorted(set(tags) | set(info.tags))
    tag_values = c2.multiselect(
        "タグ", tag_options, default=info.tags, accept_new_options=True, key=f"{key}_tags",
        placeholder="選択、または入力して Enter（例：統合報告書、FY2026）",
    )
    note = st.text_area("メモ", value=info.note or "", key=f"{key}_note", height=80)
    return DocInfo(doc_key, title, client, tag_values, note).cleaned()


st.title("RAG Translation App")
if msg := st.session_state.pop("flash", None):
    st.success(msg)

search_tab, ingest_tab, docs_tab = st.tabs(["過去翻訳を検索", "データ取り込み", "文書管理"], key="main_tab")

# =====================================================================
# 検索
# =====================================================================
with search_tab:
    query = st.text_area("英文または和文を貼り付けてください", height=150, key="query")
    c1, c2, c3 = st.columns([2, 1, 1])
    direction = c1.radio("検索の向き", list(DIRECTIONS), horizontal=True, key="direction",
                         help="自動判定：かな・漢字を含めば和文（和→英）、それ以外は英文（英→和）として検索します。")
    top_k = c2.slider("表示件数", 1, 20, 5)
    min_rate = c3.slider("最低一致率（%）", 0, 100, 0)

    with st.expander("検索対象の絞り込み（顧客名・案件名・タグ）"):
        try:
            with connect() as conn:
                all_clients, all_titles, all_tags = list_clients(conn), list_titles(conn), list_tags(conn)
        except Exception as exc:
            all_clients, all_titles, all_tags = [], [], []
            st.error(f"データベースに接続できません: {exc}")
        f1, f2 = st.columns(2)
        sel_clients = f1.multiselect("顧客名（いずれか）", all_clients, key="filter_clients")
        sel_titles = f2.multiselect("案件名（いずれか）", all_titles, key="filter_titles")
        title_keyword = f1.text_input("案件名に含む語（部分一致）", key="filter_title_keyword",
                                      placeholder="例）サスレポ")
        sel_tags = f2.multiselect("タグ", all_tags, key="filter_tags")
        tag_mode = f2.radio("タグの条件", ["すべて含む", "いずれかを含む"], horizontal=True,
                            key="filter_tag_mode")
        st.caption("未指定なら全文書が対象です。複数の条件を指定した場合は、すべての条件を満たす文書が対象です。"
                   "絞り込み時は対象文書の全行と比較するため、取りこぼしはありません。")

    if st.button("検索", type="primary"):
        if not query.strip():
            st.warning("英文または和文を入力してください。")
        else:
            lang = DIRECTIONS[direction] or detect_lang(query)
            try:
                with st.spinner("検索しています..."):
                    with connect() as conn:
                        doc_ids = None
                        if sel_clients or sel_titles or title_keyword.strip() or sel_tags:
                            doc_ids = find_document_ids(
                                conn, clients=sel_clients, titles=sel_titles,
                                title_keyword=title_keyword, tags=sel_tags,
                                tag_mode="all" if tag_mode == "すべて含む" else "any")
                        hits = hybrid_search(conn, get_embedder(), query, lang=lang,
                                             doc_ids=doc_ids, top_k=top_k)
            except Exception as exc:
                st.error(f"検索に失敗しました: {exc}")
            else:
                src, tgt = LANG_LABEL[lang], LANG_LABEL["ja" if lang == "en" else "en"]
                scope = f"・対象 {len(doc_ids)} 文書" if doc_ids is not None else ""
                st.caption(f"{src}として検索（{'英→和' if lang == 'en' else '和→英'}）{scope}")
                hits = [h for h in hits if h["match_rate"] >= min_rate]
                hits.sort(key=lambda h: (h["match_rate"], h["score"]), reverse=True)
                if not hits:
                    st.info("該当する過去翻訳が見つかりませんでした。")
                for i, h in enumerate(hits, 1):
                    page = f" p.{h['page_source']}" if h["page_source"] is not None else ""
                    label = html.escape(" ／ ".join(filter(None, [h["title"] or h["doc_key"], h["client"]])))
                    with st.container(border=True):
                        st.markdown(
                            f'<span style="font-size:1.4em;font-weight:bold;'
                            f'color:{rate_color(h["match_rate"])}">{h["match_rate"]}%</span>'
                            f"　#{i}　<small>{label}{page}"
                            f"　(score {h['score']:.3f} / trgm {h['trgm_sim']:.3f}"
                            f" / vec {h['vec_sim']:.3f})</small>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**過去の{src}**")
                        st.write(h["source_text"])
                        st.markdown(f"**過去の{tgt}（訳文）**")
                        st.write(h["target_text"])
                        st.markdown(f"**差分（過去の{src} → 今回の{src}）**")
                        if not h["diff_ops"]:
                            st.caption("差分なし")
                        else:
                            st.markdown(
                                f'<div style="line-height:1.8">{diff_html(h["diff_marked"])}</div>',
                                unsafe_allow_html=True,
                            )

# =====================================================================
# データ取り込み
# =====================================================================
with ingest_tab:
    uploaded = st.file_uploader("対訳表（.xlsx / .csv）をアップロード", type=["xlsx", "csv"],
                                key="bitext_file")
    st.caption(
        "idml2xlsx で作成した対訳表をそのまま使えます（「対訳表」シートの「日本語」「英語」列、"
        "「文書情報」シートの案件名・ファイル名など）。1ファイル＝1文書として登録します。"
    )

    if uploaded:
        token = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("bitext_token") != token:
            try:
                data = BytesIO(uploaded.getvalue())
                parsed = read_csv(data) if uploaded.name.lower().endswith(".csv") else read_excel(data)
            except ValueError as exc:
                parsed = exc
            st.session_state["bitext_token"] = token
            st.session_state["bitext_parsed"] = parsed
        parsed = st.session_state["bitext_parsed"]

        if isinstance(parsed, Exception):
            st.error(str(parsed))
        elif not parsed.rows:
            st.error("取り込める対訳がありません。英文・和文の両方が入った行を確認してください。")
        else:
            st.info(f"対訳 {len(parsed.rows):,} 行を読み込みました"
                    f"（片側のみ・空行など {parsed.skipped:,} 行は取り込みません）。")

            # 初期値：「文書情報」シート → 旧形式の doc_key / title 列 → ファイル名
            info = parsed.doc_info or DocInfo()
            if not info.doc_key:
                keys = {r["doc_key"] for r in parsed.rows if r["doc_key"]}
                info.doc_key = keys.pop() if len(keys) == 1 else Path(uploaded.name).stem
            if not info.title:
                info.title = next((str(r["title"]) for r in parsed.rows if r["title"]), None)

            try:
                with connect() as conn:
                    clients, tags = list_clients(conn), list_tags(conn)
            except Exception as exc:
                clients, tags = [], []
                st.error(f"データベースに接続できません: {exc}")

            st.subheader("文書情報")
            if parsed.doc_info:
                st.caption("「文書情報」シートの内容を初期値にしています。必要に応じて修正してください。")
            form = doc_info_inputs(f"ingest_{token}", info, clients, tags)

            existing = None
            if form.doc_key:
                try:
                    with connect() as conn:
                        existing = get_document(conn, form.doc_key)
                except Exception:
                    existing = None
            replace = False
            if existing:
                st.warning(f"ファイル名「{form.doc_key}」は登録済みです（対訳 {existing['segments']:,} 行）。"
                           "文書情報は入力した内容で更新されます。")
                replace = st.radio(
                    "登録方法", ["追加（同じ対訳は重複させない）", "置き換え（既存の対訳を削除して登録）"],
                    key=f"ingest_mode_{token}",
                ).startswith("置き換え")

            if st.button("取り込む", type="primary", disabled=not form.doc_key):
                bar = st.progress(0.0, text="開始…")
                try:
                    rows = [dict(r) for r in parsed.rows]  # 再取り込みに備えて元データは残す
                    result = ingest_bitext(
                        Bitext(rows, None, parsed.skipped), form, embedder=get_embedder(),
                        replace=replace,
                        progress=lambda frac, msg: bar.progress(min(frac, 1.0), text=msg),
                    )
                except Exception as exc:
                    bar.empty()
                    st.error(f"取り込みに失敗しました: {exc}")
                else:
                    bar.empty()
                    detail = f"、既存 {result.replaced:,} 行を削除" if result.replaced else ""
                    st.success(f"「{form.doc_key}」の取り込みが完了しました"
                               f"（新規 {result.inserted:,} 行{detail}）。")

# =====================================================================
# 文書管理
# =====================================================================
with docs_tab:
    try:
        with connect() as conn:
            docs = list_documents(conn)
            clients, tags = list_clients(conn), list_tags(conn)
    except Exception as exc:
        docs, clients, tags = [], [], []
        st.error(f"データベースに接続できません: {exc}")

    if not docs:
        st.info("登録された文書はまだありません。「データ取り込み」タブから対訳表を登録してください。")
    else:
        table = pd.DataFrame([{
            "ファイル名": d["doc_key"], "案件名": d["title"], "顧客名": d["client"],
            "タグ": "、".join(d["tags"] or []), "メモ": d["note"] or "", "対訳数": d["segments"],
            "登録日時": d["created_at"].strftime("%Y-%m-%d %H:%M"),
            "更新日時": d["updated_at"].strftime("%Y-%m-%d %H:%M"),
        } for d in docs])
        st.dataframe(table, hide_index=True)

        st.subheader("文書情報の編集")
        by_id = {d["id"]: d for d in docs}
        doc_id = st.selectbox(
            "編集する文書", list(by_id),
            format_func=lambda i: f"{by_id[i]['doc_key']}　{by_id[i]['title'] or ''}",
            key="edit_doc",
        )
        doc = by_id[doc_id]
        edited = doc_info_inputs(
            f"edit_{doc_id}",
            DocInfo(doc["doc_key"], doc["title"], doc["client"], list(doc["tags"] or []), doc["note"]),
            clients, tags,
        )
        b1, b2 = st.columns([1, 3])
        if b1.button("保存", type="primary", disabled=not edited.doc_key):
            try:
                with connect() as conn:
                    update_document(conn, doc_id, edited)
            except psycopg.errors.UniqueViolation:
                st.error(f"ファイル名「{edited.doc_key}」は別の文書で使われています。")
            except Exception as exc:
                st.error(f"保存に失敗しました: {exc}")
            else:
                flash(f"「{edited.doc_key}」の文書情報を保存しました。")
                st.rerun()

        with st.expander("この文書を削除"):
            confirm = st.checkbox(
                f"「{doc['doc_key']}」と対訳 {doc['segments']:,} 行を削除します（元に戻せません）",
                key=f"confirm_delete_{doc_id}",
            )
            if st.button("削除する", disabled=not confirm, key=f"delete_{doc_id}"):
                try:
                    with connect() as conn:
                        delete_document(conn, doc_id)
                except Exception as exc:
                    st.error(f"削除に失敗しました: {exc}")
                else:
                    flash(f"「{doc['doc_key']}」を削除しました。")
                    st.rerun()
