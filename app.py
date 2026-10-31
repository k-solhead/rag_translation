"""RAG Translation App（Streamlit）― 過去翻訳の検索、RAG 翻訳、対訳表の登録、用語集・文書の管理。"""
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
from glossary import (
    COMMON, Term, client_label, count_terms, delete_client_terms, ensure_glossary_table,
    get_terms, glossary_to_excel, list_glossary_clients, load_glossary, read_glossary_file, save_terms,
)
from glossary_extract import (
    COLUMNS, build_candidates, candidates_to_excel, candidates_to_records, extract_short_pairs,
    extract_with_llm, pairs_from_db, pairs_from_rows,
)
from ingest import Bitext, ingest_bitext, read_csv, read_excel
from llm import available_engines
from search import hybrid_search
from translation import (
    MODE_LABEL, TranslationSettings, detect_document_lang, extract_from_docx, extract_from_text,
    extract_from_xlsx, make_units, results_to_excel, run_translation,
)

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


@st.cache_resource(show_spinner=False)
def ensure_schema() -> bool:
    """既存の DB に、後から追加したテーブル（用語集）を作る。"""
    with connect() as conn:
        ensure_glossary_table(conn)
    return True


@st.cache_data(ttl=120, show_spinner="使える LLM を確認しています...")
def get_engines():
    return available_engines()


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


def doc_filter_inputs(prefix: str) -> dict:
    """顧客名・案件名・タグによる文書の絞り込み欄（検索タブ・翻訳タブで共通）。"""
    try:
        with connect() as conn:
            all_clients, all_titles, all_tags = list_clients(conn), list_titles(conn), list_tags(conn)
    except Exception as exc:
        all_clients, all_titles, all_tags = [], [], []
        st.error(f"データベースに接続できません: {exc}")
    f1, f2 = st.columns(2)
    f = {
        "clients": f1.multiselect("顧客名（いずれか）", all_clients, key=f"{prefix}_clients"),
        "titles": f2.multiselect("案件名（いずれか）", all_titles, key=f"{prefix}_titles"),
        "title_keyword": f1.text_input("案件名に含む語（部分一致）", key=f"{prefix}_title_keyword",
                                       placeholder="例）サスレポ"),
        "tags": f2.multiselect("タグ", all_tags, key=f"{prefix}_tags"),
        "tag_mode": f2.radio("タグの条件", ["すべて含む", "いずれかを含む"], horizontal=True,
                             key=f"{prefix}_tag_mode"),
    }
    st.caption("未指定なら全文書が対象です。複数の条件を指定した場合は、すべての条件を満たす文書が対象です。"
               "絞り込み時は対象文書の全行と比較するため、取りこぼしはありません。")
    return f


def resolve_doc_ids(conn, f: dict) -> list[int] | None:
    """絞り込み条件 → 対象文書の id（条件なしなら None ＝全文書）。"""
    if not (f["clients"] or f["titles"] or f["title_keyword"].strip() or f["tags"]):
        return None
    return find_document_ids(conn, clients=f["clients"], titles=f["titles"],
                             title_keyword=f["title_keyword"], tags=f["tags"],
                             tag_mode="all" if f["tag_mode"] == "すべて含む" else "any")


st.title("RAG Translation App")
if msg := st.session_state.pop("flash", None):
    st.success(msg)
try:
    ensure_schema()
except Exception as exc:
    st.error(f"データベースに接続できません: {exc}")

search_tab, translate_tab, ingest_tab, glossary_tab, docs_tab = st.tabs(
    ["過去翻訳を検索", "翻訳", "データ取り込み", "用語集", "文書管理"], key="main_tab")

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
        search_filter = doc_filter_inputs("filter")

    if st.button("検索", type="primary"):
        if not query.strip():
            st.warning("英文または和文を入力してください。")
        else:
            lang = DIRECTIONS[direction] or detect_lang(query)
            try:
                with st.spinner("検索しています..."):
                    with connect() as conn:
                        doc_ids = resolve_doc_ids(conn, search_filter)
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
# 翻訳（RAG で前年の訳を踏襲）
# =====================================================================
with translate_tab:
    st.caption(
        "RAG で抽出した前年の原文・差分・訳文を LLM に渡し、前年の訳を踏襲した訳文を作ります。"
        "複数の LLM を同時に使い、結果を Excel にまとめます。"
    )
    src_kind = st.radio("入力", ["ファイル（Word / Excel）", "テキストを貼り付け"], horizontal=True,
                        key="tr_input")
    paragraphs: list[str] = []
    input_name = ""
    if src_kind.startswith("ファイル"):
        tr_file = st.file_uploader("翻訳する Word / Excel ファイル", type=["docx", "xlsx"], key="tr_file")
        if tr_file:
            try:
                data = tr_file.getvalue()
                paragraphs = (extract_from_xlsx(data) if tr_file.name.lower().endswith(".xlsx")
                              else extract_from_docx(data))
                input_name = Path(tr_file.name).stem
            except Exception as exc:
                st.error(f"ファイルを読み込めません: {exc}")
    else:
        pasted = st.text_area("翻訳する文章（改行で段落を区切ります）", height=200, key="tr_text")
        paragraphs = extract_from_text(pasted)
        input_name = "貼り付け"
    if paragraphs:
        st.caption(f"{len(paragraphs):,} 段落を読み込みました。")

    c1, c2 = st.columns(2)
    tr_dir = c1.radio("翻訳の向き", ["自動判定", "英→和", "和→英"], horizontal=True, key="tr_dir")
    split = c2.checkbox("文単位に分けて翻訳する（推奨：過去訳との一致を見つけやすくなります）",
                        value=True, key="tr_split")

    engines, engine_warnings = get_engines()
    for w in engine_warnings:
        st.warning(w)
    by_label = {e.label: e for e in engines}
    e1, e2 = st.columns([5, 1])
    sel_labels = e1.multiselect(
        "翻訳に使う LLM（複数可・並列で実行）", list(by_label), key="tr_engines",
        help="「RAG」は前年訳・差分・用語集を使うプロンプトです。「従来プロンプト」「PLaMo翻訳」"
             "「TranslateGemma」はモデル専用の短いプロンプトで、用語集は従来プロンプトのみ使います。",
    )
    if e2.button("一覧を更新", key="tr_refresh"):
        get_engines.clear()
        st.rerun()

    with st.expander("過去訳（RAG）の検索対象 ― 前年の文書に絞り込むと精度が上がります"):
        tr_filter = doc_filter_inputs("tr_filter")

    try:
        with connect() as conn:
            glossary_clients = sorted(set(list_glossary_clients(conn)) - {COMMON})
            term_counts = count_terms(conn)
    except Exception:
        glossary_clients, term_counts = [], {}
    default_client = tr_filter["clients"][0] if len(tr_filter["clients"]) == 1 else None
    g_options = [COMMON] + glossary_clients
    g_client = st.selectbox(
        "用語集", g_options,
        index=g_options.index(default_client) if default_client in g_options else 0,
        format_func=lambda c: (f"{c}（＋共通）" if c else "共通のみ") + f"　{term_counts.get(c, 0)}語",
        key="tr_glossary",
        help="各文に含まれる用語だけを LLM に渡します。顧客の用語集を選ぶと共通の用語集も併用し、"
             "同じ用語は顧客の訳語を優先します。",
    )

    with st.expander("詳細設定"):
        d1, d2, d3 = st.columns(3)
        edit_th = d1.slider("「修正」にする一致率（%）", 50, 99, 75, key="tr_edit_th",
                            help="前年訳を差分の箇所だけ直させます。")
        ref_th = d2.slider("「参考」にする一致率（%）", 0, edit_th, min(40, edit_th), key="tr_ref_th",
                           help="過去訳を用語・文体の参考として渡し、新しく訳させます。これ未満は「新規」です。")
        ref_count = d3.slider("参考として渡す過去訳の件数", 1, 5, 3, key="tr_ref_count")
        reuse_exact = st.checkbox("完全一致（100%）の文は LLM に送らず前年訳をそのまま使う", value=True,
                                  key="tr_reuse_exact")

    if st.button("翻訳を実行", type="primary", disabled=not (paragraphs and sel_labels), key="tr_run"):
        lang = {"英→和": "en", "和→英": "ja"}.get(tr_dir) or detect_document_lang(paragraphs)
        units = make_units(paragraphs, lang, split)
        bar = st.progress(0.0, text="開始…")
        try:
            with connect() as conn:
                doc_ids = resolve_doc_ids(conn, tr_filter)
                if doc_ids == []:
                    st.warning("絞り込み条件に合う文書がないため、過去訳を使わずに翻訳します。")
                settings = TranslationSettings(
                    lang=lang, engines=[by_label[x] for x in sel_labels], doc_ids=doc_ids,
                    glossary=load_glossary(conn, g_client or None),
                    edit_threshold=edit_th, ref_threshold=ref_th, ref_count=ref_count,
                    reuse_exact=reuse_exact,
                )
                rows = run_translation(conn, get_embedder(), units, settings,
                                       progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
            meta = {
                "入力": input_name,
                "過去訳の検索対象": "全文書" if doc_ids is None else f"{len(doc_ids)} 文書（絞り込みあり）",
                "用語集": f"{g_client}（＋共通）" if g_client else "共通のみ",
            }
            data = results_to_excel(rows, settings, meta)
        except Exception as exc:
            bar.empty()
            st.error(f"翻訳に失敗しました: {exc}")
        else:
            bar.empty()
            preview = pd.DataFrame([{
                "No.": r.unit.no, "原文": r.unit.text,
                "一致率": r.top["match_rate"] if r.top else None, "処理": MODE_LABEL[r.mode],
                **{e.label: r.outputs.get(e.key, "") for e in settings.engines},
            } for r in rows])
            st.session_state["tr_result"] = {
                "data": data, "preview": preview,
                "name": f"{input_name}_翻訳_{pd.Timestamp.now():%Y%m%d_%H%M}.xlsx",
                "modes": {m: sum(r.mode == m for r in rows) for m in MODE_LABEL},
                "errors": sum(len(r.errors) for r in rows),
                "checks": sum(bool(r.checks.get(e.key)) for r in rows for e in settings.engines),
            }

    res = st.session_state.get("tr_result")
    if res:
        st.success("翻訳が完了しました。")
        m = st.columns(6)
        for col, mode in zip(m, ["exact", "edit", "reference", "new"]):
            col.metric(MODE_LABEL[mode], f"{res['modes'].get(mode, 0):,} 文")
        m[4].metric("要確認", f"{res['checks']:,} 件")
        m[5].metric("エラー", f"{res['errors']:,} 件")
        st.download_button("Excel をダウンロード", data=res["data"], file_name=res["name"],
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           type="primary", key="tr_download")
        st.dataframe(res["preview"], hide_index=True, height=400)

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
# 用語集
# =====================================================================
with glossary_tab:
    try:
        with connect() as conn:
            gl_clients = list_glossary_clients(conn)
            doc_clients = list_clients(conn)
            gl_counts = count_terms(conn)
    except Exception as exc:
        gl_clients, doc_clients, gl_counts = [], [], {}
        st.error(f"データベースに接続できません: {exc}")
    gl_options = [COMMON] + sorted((set(gl_clients) | set(doc_clients)) - {COMMON})
    gl_client = st.selectbox(
        "用語集（顧客）", gl_options,
        format_func=lambda c: f"{client_label(c)}　{gl_counts.get(c, 0)}語",
        accept_new_options=True, key="gl_client",
        placeholder="選択、または新しい顧客名を入力して Enter",
    ) or COMMON
    st.caption("「（共通）」の用語集はすべての翻訳で使われます。顧客の用語集に同じ英語の用語があれば、"
               "顧客の訳語が優先されます。区分「必須」は毎回この訳語を使う用語（自動チェックの対象）、"
               "「参考」は文脈に合えば使う訳語の候補です。")
    version = st.session_state.get("gl_version", 0)
    gl_view = st.radio("操作", ["用語集を編集", "対訳表から用語を抽出"], horizontal=True, key="gl_view",
                       label_visibility="collapsed")

    if gl_view == "用語集を編集":
        try:
            with connect() as conn:
                current_terms = get_terms(conn, gl_client)
        except Exception:
            current_terms = []
        gl_df = pd.DataFrame([{"en": t.en, "ja": t.ja, "required": t.required, "note": t.note or ""}
                              for t in current_terms], columns=["en", "ja", "required", "note"])
        edited = st.data_editor(
            gl_df, num_rows="dynamic", hide_index=True, key=f"gl_editor_{gl_client}_{version}",
            column_config={
                "en": st.column_config.TextColumn("英語（en）", required=True, width="large"),
                "ja": st.column_config.TextColumn("日本語（ja）", required=True, width="large"),
                "required": st.column_config.CheckboxColumn("必須", default=True,
                                                            help="オフにすると「参考」になります"),
                "note": st.column_config.TextColumn("備考", width="medium"),
            },
        )
        b1, b2 = st.columns([1, 4])
        if b1.button("保存", type="primary", key="gl_save",
                     help="表の内容で、この用語集を置き換えます（行の追加・修正・削除を反映）。"):
            terms = [Term(str(r["en"]), str(r["ja"]), gl_client, str(r["note"] or "") or None,
                          bool(r["required"]) if r["required"] != "" else True)
                     for r in edited.fillna("").to_dict("records")
                     if str(r["en"]).strip() and str(r["ja"]).strip()]
            try:
                with connect() as conn:
                    n = save_terms(conn, gl_client, terms, replace=True)
            except Exception as exc:
                st.error(f"保存に失敗しました: {exc}")
            else:
                st.session_state["gl_version"] = version + 1
                flash(f"{client_label(gl_client)} の用語集を保存しました（{n:,} 語）。")
                st.rerun()
        b2.download_button(
            "Excel でダウンロード", data=glossary_to_excel(current_terms),
            file_name=f"用語集_{client_label(gl_client)}.xlsx", key="gl_download",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            disabled=not current_terms,
        )

        with st.expander("ファイルから取り込む（CSV / Excel）"):
            st.caption("列名は en / ja（英語・日本語、英文・和文も可）、任意で note（備考）・区分（必須／参考）。"
                       "列名のない2列のファイル（1列目＝英語、2列目＝日本語）も読めます。"
                       "「採用」列がある場合（用語抽出の候補一覧）は、○ の行だけを取り込みます。")
            gl_file = st.file_uploader("用語集ファイル", type=["csv", "xlsx"], key=f"gl_file_{version}")
            strip_quotes = st.checkbox("用語全体を囲む「」『』“” を外す（文中の括弧は翻訳時に文脈で付けるため）",
                                       value=True, key="gl_strip_quotes")
            if gl_file:
                try:
                    imported = read_glossary_file(gl_file.getvalue(), gl_file.name, strip_quotes=strip_quotes)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    n_ref = sum(not t.required for t in imported)
                    st.info(f"{len(imported):,} 語を読み込みました（必須 {len(imported) - n_ref:,}・参考 {n_ref:,}）。"
                            "同じ英語・日本語の組は1つにまとめています。")
                    st.dataframe(pd.DataFrame([{"en": t.en, "ja": t.ja, "区分": t.kind, "note": t.note or ""}
                                               for t in imported[:20]]), hide_index=True)
                    gl_mode = st.radio("取り込み方法", ["追加・更新（今の用語は残す）", "置き換え（今の用語を削除）"],
                                       horizontal=True, key="gl_import_mode")
                    if st.button(f"{client_label(gl_client)} に取り込む", key="gl_import"):
                        try:
                            with connect() as conn:
                                n = save_terms(conn, gl_client,
                                               [Term(t.en, t.ja, gl_client, t.note, t.required) for t in imported],
                                               replace=gl_mode.startswith("置き換え"))
                        except Exception as exc:
                            st.error(f"取り込みに失敗しました: {exc}")
                        else:
                            st.session_state["gl_version"] = version + 1
                            flash(f"{client_label(gl_client)} の用語集に {n:,} 語を取り込みました。")
                            st.rerun()

        with st.expander("この用語集を削除"):
            gl_confirm = st.checkbox(f"{client_label(gl_client)} の用語 {len(current_terms):,} 語をすべて削除します",
                                     key=f"gl_confirm_{gl_client}")
            if st.button("削除する", disabled=not (gl_confirm and current_terms), key="gl_delete"):
                with connect() as conn:
                    delete_client_terms(conn, gl_client)
                st.session_state["gl_version"] = version + 1
                flash(f"{client_label(gl_client)} の用語集を削除しました。")
                st.rerun()

    else:  # 対訳表から用語を抽出
        st.caption(f"対訳から用語の候補を抽出し、{client_label(gl_client)} の用語集と照合します。"
                   "候補を確認して「採用」に印を付け、用語集に取り込みます。")
        gx_src = st.radio("対訳の取得元", ["登録済みの文書（データベース）", "対訳表ファイル（.xlsx / .csv）"],
                          horizontal=True, key="gx_src")
        gx_pairs_source = None
        if gx_src.startswith("登録済み"):
            with st.expander("対象の文書（未指定なら登録済みのすべての文書）"):
                gx_filter = doc_filter_inputs("gx_filter")
        else:
            gx_file = st.file_uploader("対訳表（idml2xlsx の出力など）", type=["xlsx", "csv"], key="gx_file")
            if gx_file:
                try:
                    data = BytesIO(gx_file.getvalue())
                    gx_pairs_source = (read_csv(data) if gx_file.name.lower().endswith(".csv")
                                       else read_excel(data)).rows
                    st.caption(f"対訳 {len(gx_pairs_source):,} 組を読み込みました。")
                except ValueError as exc:
                    st.error(str(exc))

        m1, m2 = st.columns(2)
        use_short = m1.checkbox("短い対訳から抽出（見出し・表の項目など。LLM 不要）", value=True, key="gx_short")
        max_words = m1.slider("英語の最大語数", 1, 8, 4, key="gx_max_words", disabled=not use_short)
        min_count = m1.slider("最低出現回数", 1, 10, 2, key="gx_min_count", disabled=not use_short)
        use_llm = m2.checkbox("LLM で文中の用語を抽出（固有名詞・取り組み名・専門用語など）", value=False,
                              key="gx_llm")
        gx_engines = [e for e in get_engines()[0] if e.style == "rag"]
        gx_engine_label = m2.selectbox("抽出に使う LLM", [e.label for e in gx_engines], key="gx_engine",
                                       disabled=not use_llm)
        max_pairs = m2.number_input("LLM に渡す文の上限（0＝すべて）", 0, 100000, 300, step=100, key="gx_max_pairs",
                                    disabled=not use_llm,
                                    help="30文ずつまとめて LLM に渡します。まず少なめで試すのがおすすめです。")
        include_existing = st.checkbox("既存の用語の一貫性も表示する（必須／参考の見直し用）", value=True,
                                       key="gx_include_existing")

        can_run = (use_short or (use_llm and gx_engine_label)) and \
            (gx_src.startswith("登録済み") or gx_pairs_source is not None)
        if st.button("候補を抽出", type="primary", disabled=not can_run, key="gx_run"):
            bar = st.progress(0.0, text="開始…")
            try:
                with connect() as conn:
                    if gx_src.startswith("登録済み"):
                        pairs = pairs_from_db(conn, resolve_doc_ids(conn, gx_filter))
                    else:
                        pairs = pairs_from_rows(gx_pairs_source)
                    existing = load_glossary(conn, gl_client or None)
                short = extract_short_pairs(pairs, max_words=max_words, min_count=min_count) if use_short else []
                llm_cands, llm_errors = [], []
                if use_llm and gx_engine_label:
                    engine = next(e for e in gx_engines if e.label == gx_engine_label)
                    llm_cands, llm_errors = extract_with_llm(
                        pairs, engine, max_pairs=int(max_pairs) or None,
                        progress=lambda f, m: bar.progress(0.1 + 0.7 * f, text=m))
                cands = build_candidates(pairs, short, llm_cands, existing, include_existing=include_existing,
                                         progress=lambda f, m: bar.progress(0.8 + 0.2 * f, text=m))
            except Exception as exc:
                bar.empty()
                st.error(f"抽出に失敗しました: {exc}")
            else:
                bar.empty()
                st.session_state["gx_result"] = {
                    "records": candidates_to_records(cands), "pairs": len(pairs), "errors": llm_errors,
                    "client": gl_client, "run": pd.Timestamp.now().strftime("%H%M%S"),
                }

        gx = st.session_state.get("gx_result")
        if gx:
            recs = gx["records"]
            status = pd.Series([r["既存との関係"] for r in recs]).value_counts() if recs else {}
            k = st.columns(4)
            k[0].metric("対象の対訳", f"{gx['pairs']:,} 組")
            k[1].metric("新規", f"{status.get('新規', 0):,} 語")
            k[2].metric("訳語が違う", f"{status.get('訳語が違う', 0):,} 語")
            k[3].metric("登録済み", f"{status.get('登録済み', 0):,} 語")
            for err in gx["errors"][:3]:
                st.warning(f"LLM の呼び出しで一部エラー: {err}")
            if gx["client"] != gl_client:
                st.info(f"この候補は {client_label(gx['client'])} の用語集と照合した結果です。")
            st.caption("「採用」に印を付けた行を取り込みます（初期値：新規で、一貫性80%以上かつ2回以上出現）。"
                       "一貫性＝英文に用語が出てくる対訳のうち、和文に訳語が使われている割合。"
                       "低い用語は文脈で訳し分けている可能性があるため、「参考」にするか見送ってください。"
                       "「登録済み」の行を採用すると、区分（必須／参考）だけを更新できます。")
            gx_df = pd.DataFrame(recs, columns=COLUMNS)
            gx_edited = st.data_editor(
                gx_df, hide_index=True, height=480, key=f"gx_editor_{gx['run']}_{version}",
                disabled=[c for c in COLUMNS if c not in ("採用", "en", "ja", "区分")],
                column_config={
                    "採用": st.column_config.CheckboxColumn("採用", width="small"),
                    "区分": st.column_config.SelectboxColumn("区分", options=["必須", "参考"], width="small"),
                    "en": st.column_config.TextColumn("英語（en）", width="medium"),
                    "ja": st.column_config.TextColumn("日本語（ja）", width="medium"),
                    "一貫性(%)": st.column_config.ProgressColumn("一貫性", min_value=0, max_value=100,
                                                                 format="%d%%"),
                },
            )
            adopted = gx_edited[gx_edited["採用"] == True]  # noqa: E712
            c1, c2 = st.columns([2, 3])
            if c1.button(f"採用した {len(adopted):,} 語を {client_label(gl_client)} に取り込む", type="primary",
                         disabled=adopted.empty, key="gx_import"):
                terms = [Term(str(r["en"]).strip(), str(r["ja"]).strip(), gl_client, None, r["区分"] != "参考")
                         for r in adopted.to_dict("records") if str(r["en"]).strip() and str(r["ja"]).strip()]
                try:
                    with connect() as conn:
                        n = save_terms(conn, gl_client, terms, replace=False)
                except Exception as exc:
                    st.error(f"取り込みに失敗しました: {exc}")
                else:
                    st.session_state["gl_version"] = version + 1
                    st.session_state.pop("gx_result", None)
                    flash(f"{client_label(gl_client)} の用語集に {n:,} 語を取り込みました。")
                    st.rerun()
            c2.download_button(
                "候補一覧を Excel でダウンロード（確認してから取り込む場合）",
                data=candidates_to_excel(gx_edited.to_dict("records")),
                file_name=f"用語候補_{client_label(gl_client)}.xlsx", key="gx_download",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

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
