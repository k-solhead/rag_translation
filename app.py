"""RAG Translation App（Streamlit）― 過去翻訳の検索、RAG 翻訳、対訳表の登録、用語集・文書の管理。"""
import html
import json
import re
import time
from io import BytesIO
from pathlib import Path

import pandas as pd
import psycopg
import streamlit as st

from common import LANG_LABEL, Embedder, connect, detect_lang
from auth import (
    ACTION_LABELS, AUTH_MODE, ROLES, SESSION_TIMEOUT_MINUTES, User, audit, authenticate, change_password,
    check_password_rule, create_client, create_user, delete_client, ensure_auth_tables, find_user_by_login,
    generate_password, has_users, list_audit_log, list_client_names, list_clients_detail, list_users,
    llm_forbidden_clients, load_user, record_sso_login, rename_client, reset_password, set_client_llm_allowed,
    unlock_user, update_user, verify_current_password,
)
from documents import (
    DocInfo, delete_document, document_ids_in_scope, find_document_ids, get_document, get_document_by_id,
    list_clients, list_documents, list_tags, list_titles, update_document,
)
from glossary import (
    COMMON, COMPANY_WIDE, Term, client_label, count_terms, delete_terms, ensure_glossary_table,
    get_terms, glossary_label, glossary_scope_label, glossary_to_excel, list_glossaries, load_glossary,
    read_glossary_file, save_terms,
)
from glossary_extract import (
    COLUMNS, build_candidates, candidates_to_excel, candidates_to_records, extract_short_pairs,
    extract_with_llm, pairs_from_db, pairs_from_rows,
)
from ingest import Bitext, ingest_bitext, read_csv, read_excel
from llm import available_engines
from search import hybrid_search
from usage import PROVIDERS, ensure_usage_table, format_time, get_usage, reset_usage
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
    """既存の DB に、後から追加したテーブル（用語集・トークン数の累計・利用者と権限）を作る。"""
    with connect() as conn:
        ensure_glossary_table(conn)
        ensure_usage_table(conn)
        ensure_auth_tables(conn)  # 用語集の後（既存の顧客名を顧客マスターに登録するため）
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
    client_options = list(clients)  # 登録済みで、この利用者が扱える顧客だけ（新しい顧客は管理者が登録）
    client = c1.selectbox(
        "顧客名（必須）", client_options,
        index=client_options.index(info.client) if info.client in client_options else None,
        key=f"{key}_client", placeholder="一覧から選択",
        help="顧客名は「顧客・利用者管理」タブで管理者が登録します。",
    )
    if info.client and info.client not in client_options:
        c1.warning(f"顧客名「{info.client}」は未登録か、担当外です。一覧から選ぶか、管理者に登録を依頼してください。")
    tag_options = sorted(set(tags) | set(info.tags))
    tag_values = c2.multiselect(
        "タグ", tag_options, default=info.tags, accept_new_options=True, key=f"{key}_tags",
        placeholder="選択、または入力して Enter（例：統合報告書、FY2026）",
    )
    note = st.text_area("メモ", value=info.note or "", key=f"{key}_note", height=80)
    return DocInfo(doc_key, title, client, tag_values, note).cleaned()


def doc_filter_inputs(prefix: str, scope: list[str] | None) -> dict:
    """顧客名・案件名・タグによる文書の絞り込み欄（検索タブ・翻訳タブで共通）。選択肢は見られる文書の範囲だけ。"""
    try:
        with connect() as conn:
            all_clients = list_clients(conn, scope)
            all_titles, all_tags = list_titles(conn, scope), list_tags(conn, scope)
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
    st.caption("未指定なら、見られるすべての文書が対象です。複数の条件を指定した場合は、すべての条件を満たす文書が対象です。"
               "絞り込み時は対象文書の全行と比較するため、取りこぼしはありません。")
    return f


def resolve_doc_ids(conn, f: dict, scope: list[str] | None,
                    exclude: list[str] | None = None) -> list[int] | None:
    """絞り込み条件 → 対象文書の id。条件なしなら見られる全文書（制限がなければ None ＝全文書）。

    scope   : 見られる顧客（None なら制限なし）
    exclude : 除外する顧客（社外 LLM を使わない顧客）
    """
    if not (f["clients"] or f["titles"] or f["title_keyword"].strip() or f["tags"]):
        return document_ids_in_scope(conn, scope, exclude)
    return find_document_ids(conn, clients=f["clients"], titles=f["titles"],
                             title_keyword=f["title_keyword"], tags=f["tags"],
                             tag_mode="all" if f["tag_mode"] == "すべて含む" else "any",
                             scope=scope, exclude_clients=exclude)


def show_target_docs(f: dict | None, heading: str, empty_note: str, scope: list[str] | None,
                     exclude: list[str] | None = None) -> list[int] | None:
    """絞り込みの結果（条件がなければ全文書）を枠内に常に表示し、対象文書の id を返す。

    f=None は「検索対象文書なし」（[] を返す）。DB に接続できないときは None。
    """
    with st.container(border=True):
        if f is None:
            st.markdown(f"**{heading}：なし**（過去訳を検索せず、用語集だけで翻訳します）")
            st.caption("すべての文が「新規」として LLM に送られます。"
                       "「PLaMo翻訳」「TranslateGemma」は用語集を使わないため、原文だけで翻訳します。")
            return []
        try:
            with connect() as conn:
                doc_ids = resolve_doc_ids(conn, f, scope, exclude)
                all_docs = list_documents(conn, scope)
        except Exception as exc:
            st.error(f"データベースに接続できません: {exc}")
            return None
        if doc_ids is None:
            target_docs = all_docs
        else:
            id_set = set(doc_ids)
            target_docs = [d for d in all_docs if d["id"] in id_set]
        n_pairs = sum(d["segments"] for d in target_docs)
        is_all = not (f["clients"] or f["titles"] or f["title_keyword"].strip() or f["tags"])
        if is_all:
            label = "全文書" if scope is None else "担当顧客の全文書"
            st.markdown(f"**{heading}：{label}**（{len(target_docs):,} 文書・対訳 {n_pairs:,} 組）")
        elif not target_docs:
            st.markdown(f"**{heading}：該当する文書がありません**")
            st.caption(empty_note)
        else:
            st.markdown(f"**{heading}：絞り込み結果 {len(target_docs):,} 文書**（対訳 {n_pairs:,} 組）")
        if exclude:
            n_ex = sum(1 for d in all_docs if d["client"] in exclude)
            if n_ex:
                st.caption(f"「社外 LLM を使わない」設定の顧客の文書（{n_ex:,} 件）は、過去訳として LLM に渡さないため対象外です。")
        if target_docs:
            st.dataframe(
                pd.DataFrame([{
                    "ファイル名": d["doc_key"], "案件名": d["title"] or "", "顧客名": d["client"] or "",
                    "タグ": "、".join(d["tags"] or []), "対訳数": d["segments"],
                } for d in target_docs]),
                hide_index=True, height=min(35 * (len(target_docs) + 1) + 3, 220),
            )
    return doc_ids


# =====================================================================
# ログイン
# =====================================================================
def _clear_login() -> None:
    for k in ("auth_uid", "auth_last", "auth_sso_recorded"):
        st.session_state.pop(k, None)


def _logout() -> None:
    uid = st.session_state.get("auth_uid")
    if uid:
        try:
            with connect() as conn:
                audit(conn, load_user(conn, uid), "logout")
        except Exception:
            pass
    _clear_login()
    if AUTH_MODE == "oidc":
        st.logout()


def _password_form(user: User, *, forced: bool) -> None:
    """パスワードの変更欄（初回ログイン時は必須、サイドバーからは任意）。"""
    with st.form(f"pw_form_{'forced' if forced else 'self'}", clear_on_submit=True):
        if forced:
            st.info("初めてのログイン、またはパスワードの再設定後です。新しいパスワードを設定してください。")
        current = st.text_input("今のパスワード", type="password")
        new1 = st.text_input("新しいパスワード（10 文字以上・英大文字／英小文字／数字のうち 2 種類以上）",
                             type="password")
        new2 = st.text_input("新しいパスワード（確認）", type="password")
        if st.form_submit_button("パスワードを変更", type="primary"):
            with connect() as conn:
                if not verify_current_password(conn, user.id, current):
                    st.error("今のパスワードが違います。")
                elif new1 != new2:
                    st.error("新しいパスワードが一致しません。")
                elif new1 == current:
                    st.error("今と違うパスワードにしてください。")
                elif msg := check_password_rule(new1):
                    st.error(msg)
                else:
                    change_password(conn, user.id, new1)
                    audit(conn, user, "password_change")
                    conn.commit()
                    flash("パスワードを変更しました。")
                    st.rerun()


def require_login() -> User:
    """ログイン済みの利用者を返す。未ログインならログイン画面を表示して、ここで処理を止める。

    利用者は毎回データベースから読み直す（役割・担当顧客の変更や無効化をすぐ反映するため）。
    """
    now = time.time()
    uid = st.session_state.get("auth_uid")
    if uid:
        last = st.session_state.get("auth_last", now)
        if now - last > SESSION_TIMEOUT_MINUTES * 60:
            with connect() as conn:
                audit(conn, load_user(conn, uid), "session_timeout")
            _clear_login()
            st.session_state["auth_msg"] = (f"{SESSION_TIMEOUT_MINUTES} 分間操作がなかったため、ログアウトしました。"
                                            "もう一度ログインしてください。")
            if AUTH_MODE == "oidc":
                st.logout()
        else:
            with connect() as conn:
                user = load_user(conn, uid)
            if user is None:
                _clear_login()
                st.session_state["auth_msg"] = "この利用者は利用できなくなりました。管理者にお問い合わせください。"
            else:
                st.session_state["auth_last"] = now
                if user.must_change_password and AUTH_MODE != "oidc":
                    st.title("RAG Translation App")
                    st.subheader(f"{user.display_name} さん")
                    _password_form(user, forced=True)
                    st.button("ログアウト", on_click=_logout)
                    st.stop()
                return user

    st.title("RAG Translation App")
    if msg := st.session_state.pop("auth_msg", None):
        st.warning(msg)

    if AUTH_MODE == "oidc":  # 社内アカウントでのシングルサインオン
        try:
            sso_ready = "auth" in st.secrets
        except Exception:  # secrets.toml が無い
            sso_ready = False
        if not sso_ready:
            st.error("シングルサインオンの設定（.streamlit/secrets.toml の [auth]）がありません。"
                     "README の「社内アカウントでのシングルサインオン」を参照して設定するか、"
                     ".env を AUTH_MODE=local に戻してください。")
            st.stop()
        if not st.user.get("is_logged_in", False):
            st.button("社内アカウントでログイン", type="primary", on_click=st.login)
            st.stop()
        email = st.user.get("email") or st.user.get("preferred_username") or ""
        with connect() as conn:
            user = find_user_by_login(conn, email)
            if user is None:
                audit(conn, None, "login_failed", login_id=email, detail={"reason": "not_registered", "method": "oidc"})
                conn.commit()
        if user is None:
            st.error(f"{email} はこのアプリの利用者として登録されていません。管理者に登録を依頼してください。")
            st.button("ログアウト", on_click=st.logout)
            st.stop()
        with connect() as conn:
            record_sso_login(conn, user)
        st.session_state.update(auth_uid=user.id, auth_last=now)
        st.rerun()

    with connect() as conn:
        users_exist = has_users(conn)
    if not users_exist:
        st.error("利用者が登録されていません。.env に INITIAL_ADMIN_LOGIN と INITIAL_ADMIN_PASSWORD を設定して"
                 "アプリを再起動すると、最初の管理者が作成されます。")
        st.stop()
    with st.form("login_form"):
        login_id = st.text_input("ログイン ID（社内メールアドレス）")
        password = st.text_input("パスワード", type="password")
        submitted = st.form_submit_button("ログイン", type="primary")
    if submitted:
        with connect() as conn:
            user, message = authenticate(conn, login_id, password)
        if user:
            st.session_state.update(auth_uid=user.id, auth_last=now)
            st.rerun()
        st.error(message)
    st.stop()


st.title("RAG Translation App")
if msg := st.session_state.pop("flash", None):
    st.success(msg)
try:
    ensure_schema()
except Exception as exc:
    st.error(f"データベースに接続できません: {exc}")
    st.stop()

USER = require_login()
with connect() as _conn:
    SCOPE = USER.visible_clients(_conn)          # 見られる顧客（None ＝制限なし：管理者）
    FORBIDDEN = llm_forbidden_clients(_conn)     # 社外 LLM を使わない顧客
    REGISTERED_CLIENTS = list_client_names(_conn)
MY_CLIENTS = REGISTERED_CLIENTS if SCOPE is None else [c for c in REGISTERED_CLIENTS if c in SCOPE]

with st.sidebar:
    st.markdown(f"**{USER.display_name}**　{USER.role_label}")
    st.caption(USER.login_id)
    if USER.is_admin or USER.all_clients:
        st.caption("担当顧客：全顧客")
    else:
        st.caption("担当顧客：" + ("、".join(USER.clients) if USER.clients else "なし（管理者に設定を依頼してください）"))
    st.button("ログアウト", on_click=_logout, key="logout")
    if AUTH_MODE != "oidc":
        with st.expander("パスワードの変更"):
            _password_form(USER, forced=False)
    st.caption(f"{SESSION_TIMEOUT_MINUTES} 分間操作がないと自動でログアウトします。")

# 役割に応じてタブを作る（権限のないタブは作らない：Streamlit は選ばれていないタブの中身も実行するため）
_tab_names = ["過去翻訳を検索", "翻訳"] + (["データ取り込み"] if USER.can_edit else []) + ["用語集"] \
    + (["文書管理"] if USER.can_edit else []) + ["トークン数管理"] \
    + (["顧客・利用者管理", "操作ログ"] if USER.is_admin else [])
_tabs = dict(zip(_tab_names, st.tabs(_tab_names, key="main_tab")))
search_tab, translate_tab, glossary_tab, usage_tab = (
    _tabs["過去翻訳を検索"], _tabs["翻訳"], _tabs["用語集"], _tabs["トークン数管理"])
ingest_tab, docs_tab = _tabs.get("データ取り込み"), _tabs.get("文書管理")
admin_tab, log_tab = _tabs.get("顧客・利用者管理"), _tabs.get("操作ログ")

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
        search_filter = doc_filter_inputs("filter", SCOPE)
    # 絞り込みの結果（条件がなければ全文書）を常に表示する
    show_target_docs(search_filter, "検索対象",
                     "このまま検索しても結果は表示されません。絞り込み条件を見直してください。", SCOPE)

    if st.button("検索", type="primary"):
        if not query.strip():
            st.warning("英文または和文を入力してください。")
        else:
            lang = DIRECTIONS[direction] or detect_lang(query)
            try:
                with st.spinner("検索しています..."):
                    with connect() as conn:
                        doc_ids = resolve_doc_ids(conn, search_filter, SCOPE)
                        hits = hybrid_search(conn, get_embedder(), query, lang=lang,
                                             doc_ids=doc_ids, top_k=top_k)
            except Exception as exc:
                st.error(f"検索に失敗しました: {exc}")
            else:
                src, tgt = LANG_LABEL[lang], LANG_LABEL["ja" if lang == "en" else "en"]
                scope_note = f"・対象 {len(doc_ids)} 文書" if doc_ids is not None else ""
                st.caption(f"{src}として検索（{'英→和' if lang == 'en' else '和→英'}）{scope_note}")
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
        help="「RAG」は前年訳・差分・用語集を使うプロンプトです。「PLaMo翻訳」「TranslateGemma」は"
             "モデル専用の形式で前年訳を使います（用語集は使いません）。「従来プロンプト」は前年訳を使わず、"
             "必須の用語だけを使います。",
    )
    if e2.button("一覧を更新", key="tr_refresh"):
        get_engines.clear()
        st.rerun()

    tr_scope = st.radio(
        "過去訳（RAG）の検索", ["検索する", "検索対象文書なし（過去訳を使わず、用語集だけで翻訳）"],
        horizontal=True, key="tr_scope",
        help="「検索対象文書なし」にすると前年訳・過去訳を一切使わず、用語集だけを渡して翻訳します。",
    )
    no_rag = tr_scope != "検索する"
    # 「検索対象文書なし」でも絞り込み欄は残す（切り替えても条件が消えず、用語集の初期値にも使うため）
    with st.expander("過去訳（RAG）の検索対象 ― 前年の文書に絞り込むと精度が上がります"
                     + ("（「検索対象文書なし」のため使いません）" if no_rag else "")):
        tr_filter = doc_filter_inputs("tr_filter", SCOPE)
    # 絞り込みの結果（条件がなければ全文書／検索しない場合は「なし」）を常に表示する
    show_target_docs(None if no_rag else tr_filter, "過去訳の検索対象",
                     "このまま翻訳すると、過去訳を使わずに翻訳します。絞り込み条件を見直してください。",
                     SCOPE, FORBIDDEN)

    try:
        with connect() as conn:
            g_glossaries = list_glossaries(conn)
            term_counts = count_terms(conn)
    except Exception:
        g_glossaries, term_counts = [], {}
    # 選べるのは担当顧客だけ（用語集がなくても、原稿の顧客として選べるようにする）
    glossary_clients = sorted(set(MY_CLIENTS))
    f_clients = tr_filter["clients"]
    default_client = f_clients[0] if len(f_clients) == 1 else None
    g_options = [COMMON] + glossary_clients
    g1, g2 = st.columns(2)
    g_client = g1.selectbox(
        "原稿の顧客（用語集）", g_options,
        index=g_options.index(default_client) if default_client in g_options else 0,
        format_func=lambda c: ((f"{c}（＋共通）" if c else "指定しない（共通の用語集のみ）")
                               + f"　{term_counts.get((c, ''), 0)}語" + ("　※社外 LLM 不可" if c in FORBIDDEN else "")),
        key=f"tr_glossary_{default_client}",  # 絞り込みの顧客が変わったら初期値を選び直す
        help="翻訳する原稿の顧客を選んでください。その顧客の用語集と共通の用語集を使い、"
             "同じ用語は顧客の訳語を優先します。「社外 LLM を使わない」設定の顧客は翻訳できません。",
    )
    llm_blocked = g_client in FORBIDDEN
    if llm_blocked:
        st.error(f"顧客「{g_client}」は「社外 LLM を使わない」設定のため、このアプリでは翻訳できません"
                 "（過去翻訳の検索は使えます）。")
    g_media = [m for c, m in g_glossaries if c == g_client and m]
    f_tags = tr_filter["tags"]
    default_medium = next((m for m in g_media if m in f_tags), COMPANY_WIDE)  # タグと同じ名前の媒体を初期値に
    m_options = [COMPANY_WIDE] + g_media
    g_medium = g2.selectbox(
        "媒体", m_options, index=m_options.index(default_medium),
        format_func=lambda m: f"{m}（＋会社全体）　{term_counts.get((g_client, m), 0)}語" if m
        else "指定しない（会社全体の用語集）",
        key=f"tr_medium_{g_client}_{default_medium}", disabled=not g_media,
        help="媒体別の用語集がある顧客だけ選べます。媒体を選ぶと「共通＋会社全体＋媒体」の用語集を重ねて使い、"
             "同じ用語は 媒体 ＞ 会社 ＞ 共通 の順に優先します。",
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

    if st.button("翻訳を実行", type="primary", disabled=not (paragraphs and sel_labels) or llm_blocked,
                 key="tr_run"):
        lang = {"英→和": "en", "和→英": "ja"}.get(tr_dir) or detect_document_lang(paragraphs)
        units = make_units(paragraphs, lang, split)
        bar = st.progress(0.0, text="開始…")
        try:
            with connect() as conn:
                if g_client in llm_forbidden_clients(conn):  # 画面表示後に設定が変わった場合に備えて再確認
                    raise ValueError(f"顧客「{g_client}」は「社外 LLM を使わない」設定です。")
                doc_ids = [] if no_rag else resolve_doc_ids(conn, tr_filter, SCOPE, llm_forbidden_clients(conn))
                if doc_ids == [] and not no_rag:
                    st.warning("絞り込み条件に合う文書がないため、過去訳を使わずに翻訳します。")
                settings = TranslationSettings(
                    lang=lang, engines=[by_label[x] for x in sel_labels], doc_ids=doc_ids,
                    glossary=load_glossary(conn, g_client or None, g_medium or None),
                    edit_threshold=edit_th, ref_threshold=ref_th, ref_count=ref_count,
                    reuse_exact=reuse_exact,
                )
                rows = run_translation(conn, get_embedder(), units, settings,
                                       progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
                audit(conn, USER, "translate", target=input_name, detail={
                    "units": len(units), "engines": [e.label for e in settings.engines],
                    "glossary": glossary_scope_label(g_client, g_medium),
                    "rag_docs": "none" if no_rag else ("all" if doc_ids is None else len(doc_ids)),
                    "modes": {m: sum(r.mode == m for r in rows) for m in MODE_LABEL}})
            meta = {
                "入力": input_name,
                "過去訳の検索対象": ("なし（用語集だけで翻訳）" if no_rag else "全文書" if doc_ids is None
                                     else f"{len(doc_ids)} 文書"),
                "実行者": f"{USER.display_name}（{USER.login_id}）",
                "用語集": glossary_scope_label(g_client, g_medium),
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
if ingest_tab is not None:  # 編集者・管理者だけ
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
                        clients, tags = MY_CLIENTS, list_tags(conn, SCOPE)
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
                out_of_scope = bool(existing) and not (SCOPE is None or existing["client"] in SCOPE)
                if out_of_scope:
                    st.error(f"ファイル名「{form.doc_key}」は別の文書で使われています。別のファイル名にしてください。")
                elif existing:
                    st.warning(f"ファイル名「{form.doc_key}」は登録済みです（対訳 {existing['segments']:,} 行）。"
                               "文書情報は入力した内容で更新されます。")
                    replace = st.radio(
                        "登録方法", ["追加（同じ対訳は重複させない）", "置き換え（既存の対訳を削除して登録）"],
                        key=f"ingest_mode_{token}",
                    ).startswith("置き換え")

                if st.button("取り込む", type="primary", disabled=not (form.doc_key and form.client) or out_of_scope):
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
                        with connect() as conn:
                            audit(conn, USER, "ingest", target=form.doc_key, detail={
                                "client": form.client, "inserted": result.inserted, "replaced": result.replaced})
                        detail = f"、既存 {result.replaced:,} 行を削除" if result.replaced else ""
                        st.success(f"「{form.doc_key}」の取り込みが完了しました"
                                   f"（新規 {result.inserted:,} 行{detail}）。")

# =====================================================================
# 用語集
# =====================================================================
with glossary_tab:
    try:
        with connect() as conn:
            gl_glossaries = [g for g in list_glossaries(conn) if g[0] == COMMON or g[0] in MY_CLIENTS]
            gl_counts = count_terms(conn)
    except Exception as exc:
        gl_glossaries, gl_counts = [], {}
        st.error(f"データベースに接続できません: {exc}")
    gl_options = [COMMON] + sorted(MY_CLIENTS)  # 共通＋担当顧客（新しい顧客は管理者が登録）

    def _client_option(c: str) -> str:
        n_media = sum(1 for cc, m in gl_glossaries if cc == c and m)
        extra = f"（媒体別 {n_media}）" if n_media else ""
        return f"{client_label(c)}　{gl_counts.get((c, COMPANY_WIDE), 0)}語{extra}"

    k1, k2 = st.columns(2)
    gl_client = (k1.selectbox(
        "用語集（顧客）", gl_options, format_func=_client_option, key="gl_client",
    ) or COMMON).strip()
    # 編集できるか：共通は管理者だけ、顧客の用語集は編集者・管理者（担当顧客のみ）
    gl_can_edit = USER.is_admin or (USER.can_edit and gl_client != COMMON and gl_client in MY_CLIENTS)
    gl_media = [m for c, m in gl_glossaries if c == gl_client and m]
    if gl_client == COMMON:  # 共通の用語集には媒体はない
        k2.selectbox("媒体", ["（共通の用語集には媒体はありません）"], disabled=True, key="gl_medium_common")
        gl_medium = COMPANY_WIDE
    else:
        gl_medium = (k2.selectbox(
            "媒体", [COMPANY_WIDE] + gl_media,
            format_func=lambda m: f"{m}　{gl_counts.get((gl_client, m), 0)}語" if m else "指定しない（会社全体）",
            accept_new_options=gl_can_edit, key=f"gl_medium_{gl_client}",
            placeholder="選択、または新しい媒体名を入力して Enter",
            help="特定の媒体（統合報告書など）だけ訳語を変える場合に使います。",
        ) or COMPANY_WIDE).strip()
    gl_label = glossary_label(gl_client, gl_medium)
    st.caption("用語集は「共通 → 会社 → 媒体」の 3 段階です。「（共通）」はすべての翻訳で使われ、"
               "同じ英語の用語は 媒体 ＞ 会社 ＞ 共通 の順に優先されます。"
               "区分「必須」は毎回この訳語を使う用語（自動チェックの対象）、"
               "「参考」は文脈に合えば使う訳語の候補です。")
    if gl_medium:
        st.info(f"媒体「{gl_medium}」の用語集です。翻訳では「共通＋{gl_client}（会社全体）＋この媒体」を重ねて使います。"
                "会社全体の用語集と訳語が違う用語だけを登録すれば十分です。"
                "媒体名は、文書のタグと同じ名前にしておくと、翻訳タブで自動的に選ばれます。")
    version = st.session_state.get("gl_version", 0)
    if gl_can_edit:
        gl_view = st.radio("操作", ["用語集を編集", "対訳表から用語を抽出"], horizontal=True, key="gl_view",
                           label_visibility="collapsed")
    else:
        gl_view = "用語集を編集"
        st.caption("この用語集は閲覧だけできます（編集は"
                   + ("管理者" if gl_client == COMMON else "編集者・管理者") + "が行います）。")

    if gl_view == "用語集を編集":
        try:
            with connect() as conn:
                current_terms = get_terms(conn, gl_client, gl_medium)
        except Exception:
            current_terms = []
        gl_df = pd.DataFrame([{"en": t.en, "ja": t.ja, "required": t.required, "note": t.note or ""}
                              for t in current_terms], columns=["en", "ja", "required", "note"])
        edited = st.data_editor(
            gl_df, num_rows="dynamic" if gl_can_edit else "fixed", hide_index=True, disabled=not gl_can_edit,
            key=f"gl_editor_{gl_client}_{gl_medium}_{version}",
            column_config={
                "en": st.column_config.TextColumn("英語（en）", required=True, width="large"),
                "ja": st.column_config.TextColumn("日本語（ja）", required=True, width="large"),
                "required": st.column_config.CheckboxColumn("必須", default=True,
                                                            help="オフにすると「参考」になります"),
                "note": st.column_config.TextColumn("備考", width="medium"),
            },
        )
        b1, b2 = st.columns([1, 4])
        if gl_can_edit and b1.button("保存", type="primary", key="gl_save",
                     help="表の内容で、この用語集を置き換えます（行の追加・修正・削除を反映）。"):
            terms = [Term(str(r["en"]), str(r["ja"]), gl_client, str(r["note"] or "") or None,
                          bool(r["required"]) if r["required"] != "" else True)
                     for r in edited.fillna("").to_dict("records")
                     if str(r["en"]).strip() and str(r["ja"]).strip()]
            try:
                with connect() as conn:
                    n = save_terms(conn, gl_client, terms, replace=True, medium=gl_medium)
                audit(conn, USER, "glossary_save", target=gl_label, detail={"terms": n})
            except Exception as exc:
                st.error(f"保存に失敗しました: {exc}")
            else:
                st.session_state["gl_version"] = version + 1
                flash(f"{gl_label} の用語集を保存しました（{n:,} 語）。")
                st.rerun()
        b2.download_button(
            "Excel でダウンロード", data=glossary_to_excel(current_terms),
            file_name=f"用語集_{gl_label}.xlsx", key="gl_download",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            disabled=not current_terms,
        )

        if gl_can_edit:  # 取り込み・削除は編集できる人だけ
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
                        if st.button(f"{gl_label} に取り込む", key="gl_import"):
                            try:
                                with connect() as conn:
                                    n = save_terms(conn, gl_client,
                                                   [Term(t.en, t.ja, gl_client, t.note, t.required) for t in imported],
                                                   replace=gl_mode.startswith("置き換え"), medium=gl_medium)
                                    audit(conn, USER, "glossary_import", target=gl_label,
                                          detail={"terms": n, "replace": gl_mode.startswith("置き換え")})
                            except Exception as exc:
                                st.error(f"取り込みに失敗しました: {exc}")
                            else:
                                st.session_state["gl_version"] = version + 1
                                flash(f"{gl_label} の用語集に {n:,} 語を取り込みました。")
                                st.rerun()

            with st.expander("この用語集を削除"):
                gl_confirm = st.checkbox(f"{gl_label} の用語 {len(current_terms):,} 語をすべて削除します",
                                         key=f"gl_confirm_{gl_client}_{gl_medium}")
                if st.button("削除する", disabled=not (gl_confirm and current_terms and gl_can_edit), key="gl_delete"):
                    with connect() as conn:
                        delete_terms(conn, gl_client, gl_medium)
                        audit(conn, USER, "glossary_delete", target=gl_label, detail={"terms": len(current_terms)})
                    st.session_state["gl_version"] = version + 1
                    flash(f"{gl_label} の用語集を削除しました。")
                    st.rerun()

    else:  # 対訳表から用語を抽出
        st.caption(f"対訳から用語の候補を抽出し、{gl_label} の用語集と照合します。"
                   "候補を確認して「採用」に印を付け、用語集に取り込みます。")
        gx_src = st.radio("対訳の取得元", ["登録済みの文書（データベース）", "対訳表ファイル（.xlsx / .csv）"],
                          horizontal=True, key="gx_src")
        gx_pairs_source = None
        if gx_src.startswith("登録済み"):
            with st.expander("対象の文書（未指定なら登録済みのすべての文書）"):
                gx_filter = doc_filter_inputs("gx_filter", SCOPE)
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
        gx_llm_blocked = gl_client in FORBIDDEN
        use_llm = m2.checkbox("LLM で文中の用語を抽出（固有名詞・取り組み名・専門用語など）", value=False,
                              key="gx_llm", disabled=gx_llm_blocked,
                              help="「社外 LLM を使わない」設定の顧客の文書は、LLM に渡しません。") and not gx_llm_blocked
        if gx_llm_blocked:
            m2.caption(f"顧客「{gl_client}」は「社外 LLM を使わない」設定のため、LLM での抽出は使えません。")
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
                        pairs = pairs_from_db(conn, resolve_doc_ids(
                            conn, gx_filter, SCOPE, llm_forbidden_clients(conn) if use_llm else None))
                    else:
                        pairs = pairs_from_rows(gx_pairs_source)
                    existing = load_glossary(conn, gl_client or None, gl_medium or None)
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
                    "label": gl_label, "run": pd.Timestamp.now().strftime("%H%M%S"),
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
            if gx["label"] != gl_label:
                st.info(f"この候補は {gx['label']} の用語集と照合した結果です。")
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
            if c1.button(f"採用した {len(adopted):,} 語を {gl_label} に取り込む", type="primary",
                         disabled=adopted.empty, key="gx_import"):
                terms = [Term(str(r["en"]).strip(), str(r["ja"]).strip(), gl_client, None, r["区分"] != "参考")
                         for r in adopted.to_dict("records") if str(r["en"]).strip() and str(r["ja"]).strip()]
                try:
                    with connect() as conn:
                        n = save_terms(conn, gl_client, terms, replace=False, medium=gl_medium)
                        audit(conn, USER, "glossary_extract_import", target=gl_label, detail={"terms": n})
                except Exception as exc:
                    st.error(f"取り込みに失敗しました: {exc}")
                else:
                    st.session_state["gl_version"] = version + 1
                    st.session_state.pop("gx_result", None)
                    flash(f"{gl_label} の用語集に {n:,} 語を取り込みました。")
                    st.rerun()
            c2.download_button(
                "候補一覧を Excel でダウンロード（確認してから取り込む場合）",
                data=candidates_to_excel(gx_edited.to_dict("records")),
                file_name=f"用語候補_{gl_label}.xlsx", key="gx_download",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# =====================================================================
# 文書管理
# =====================================================================
if docs_tab is not None:  # 編集者・管理者だけ
    with docs_tab:
        try:
            with connect() as conn:
                docs = list_documents(conn, SCOPE)
                clients, tags = MY_CLIENTS, list_tags(conn, SCOPE)
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
            if b1.button("保存", type="primary", disabled=not (edited.doc_key and edited.client)):
                try:
                    with connect() as conn:
                        cur_doc = get_document_by_id(conn, doc_id)
                        if cur_doc is None or not USER.can_see_client(conn, cur_doc["client"]) \
                                or edited.client not in MY_CLIENTS:
                            raise PermissionError("この文書を変更する権限がありません。")
                        update_document(conn, doc_id, edited)
                        audit(conn, USER, "update_document", target=edited.doc_key, detail={
                            "before": {k: cur_doc[k] for k in ("doc_key", "title", "client")},
                            "after": {"doc_key": edited.doc_key, "title": edited.title, "client": edited.client}})
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
                            cur_doc = get_document_by_id(conn, doc_id)
                            if cur_doc is None or not USER.can_see_client(conn, cur_doc["client"]):
                                raise PermissionError("この文書を削除する権限がありません。")
                            delete_document(conn, doc_id)
                            audit(conn, USER, "delete_document", target=doc["doc_key"],
                                  detail={"client": doc["client"], "segments": doc["segments"]})
                    except Exception as exc:
                        st.error(f"削除に失敗しました: {exc}")
                    else:
                        flash(f"「{doc['doc_key']}」を削除しました。")
                        st.rerun()


# =====================================================================
# トークン数管理
# =====================================================================
def _reset_provider_usage(provider: str) -> None:
    """「ゼロに初期化」ボタンの処理（確認のチェックも外す）。

    ボタンの処理（コールバック）は画面の先頭のログイン確認より先に動くため、ここでも権限を確かめる。
    """
    try:
        with connect() as conn:
            uid = st.session_state.get("auth_uid")
            actor = load_user(conn, uid) if uid else None
            if actor is None or not actor.is_admin:
                raise PermissionError("管理者だけが初期化できます。")
            reset_usage(conn, provider)
            audit(conn, actor, "usage_reset", target=provider)
            conn.commit()
    except Exception as exc:
        st.session_state["usage_error"] = f"初期化に失敗しました: {exc}"
    else:
        flash(f"{PROVIDERS[provider]} のトークン数をゼロに初期化しました。")
    st.session_state[f"usage_confirm_{provider}"] = False


with usage_tab:
    st.caption(
        "Claude・OpenAI・Gemini を呼び出すたびに、応答に含まれるトークン数を加算しています"
        "（「翻訳」と、用語集の「LLM で文中の用語を抽出」）。データベースに保存しているため、"
        "再起動しても消えず、アプリを使う全員の使用分が合算されます。"
    )
    if st.button("最新の数に更新", key="usage_refresh"):
        st.rerun()
    if err := st.session_state.pop("usage_error", None):
        st.error(err)
    try:
        with connect() as conn:
            usage_rows = get_usage(conn)
    except Exception as exc:
        usage_rows = {}
        st.error(f"データベースに接続できません: {exc}")

    usage_cols = st.columns(len(PROVIDERS))
    for col, (provider, name) in zip(usage_cols, PROVIDERS.items()):
        u = usage_rows.get(provider)
        if u is None:
            continue
        with col.container(border=True):
            st.subheader(name)
            st.metric("入力トークン（キャッシュ以外）", f"{u['input_tokens']:,}")
            st.metric("出力トークン", f"{u['output_tokens']:,}")
            st.metric("キャッシュ入力トークン（読み込み）", f"{u['cache_read_tokens']:,}")
            st.metric("キャッシュ出力トークン（書き込み）", f"{u['cache_write_tokens']:,}")
            st.caption(f"呼び出し回数 {u['requests']:,} 回\n\n最終更新 {format_time(u['updated_at'])}"
                       f"\n\n集計開始 {format_time(u['reset_at'])}")
            if USER.is_admin:
                confirm = st.checkbox(f"{name} の累計をゼロに戻す", key=f"usage_confirm_{provider}")
                st.button("ゼロに初期化", key=f"usage_reset_{provider}", disabled=not confirm,
                          on_click=_reset_provider_usage, args=(provider,))

    st.caption(
        "入力トークン × 入力単価 ＋ 出力トークン × 出力単価 ＋ キャッシュ入力 × キャッシュ読み込み単価 ＋ "
        "キャッシュ出力 × キャッシュ書き込み単価 が、おおよその料金です（単価はモデルごとに各社の料金表を参照）。"
        "OpenAI の出力トークンには推論に使った分も含まれます。OpenAI・Gemini は、キャッシュへの書き込み数が"
        "応答に含まれない場合 0 と表示されます。"
    )


# =====================================================================
# 顧客・利用者管理（管理者だけ）
# =====================================================================
if admin_tab is not None:
    with admin_tab:
        adm_view = st.radio("管理する対象", ["顧客", "利用者"], horizontal=True, key="adm_view",
                            label_visibility="collapsed")

        if adm_view == "顧客":
            st.caption("顧客名の正式な一覧です。文書・用語集・利用者の担当顧客は、ここに登録した顧客名から選びます。"
                       "表記が揺れている顧客は「名前の変更・統合」で正しい顧客名を入力すると、文書・用語集・担当者が"
                       "まとめて付け替わります。")
            try:
                with connect() as conn:
                    cl_rows = list_clients_detail(conn)
            except Exception as exc:
                cl_rows = []
                st.error(f"データベースに接続できません: {exc}")
            if cl_rows:
                st.dataframe(pd.DataFrame([{
                    "顧客名": c["name"], "社外 LLM": "使える" if c["llm_allowed"] else "使わない",
                    "文書数": c["documents"], "用語数": c["terms"], "担当者数": c["users"],
                } for c in cl_rows]), hide_index=True)

            with st.expander("顧客を追加"):
                with st.form("client_add", clear_on_submit=True):
                    new_name = st.text_input("顧客名（正式名称）", placeholder="例）アサヒグループホールディングス")
                    new_llm = st.checkbox("社外 LLM（Claude・OpenAI・Gemini）での翻訳を許可する", value=True)
                    if st.form_submit_button("追加", type="primary"):
                        try:
                            with connect() as conn:
                                name = create_client(conn, new_name, new_llm)
                                audit(conn, USER, "client_create", target=name, detail={"llm_allowed": new_llm})
                        except ValueError as exc:
                            st.error(str(exc))
                        else:
                            flash(f"顧客「{name}」を追加しました。")
                            st.rerun()

            if cl_rows:
                st.subheader("顧客の設定")
                by_name = {c["name"]: c for c in cl_rows}
                cname = st.selectbox("顧客", list(by_name), key="adm_client")
                c = by_name[cname]
                llm_ok = st.checkbox("社外 LLM（Claude・OpenAI・Gemini）での翻訳を許可する", value=c["llm_allowed"],
                                     key=f"adm_llm_{cname}",
                                     help="オフにすると、この顧客を選んだ翻訳と LLM での用語抽出ができなくなり、"
                                          "この顧客の文書は過去訳として LLM に渡されなくなります。")
                if llm_ok != c["llm_allowed"] and st.button("社外 LLM の設定を保存", key=f"adm_llm_save_{cname}"):
                    with connect() as conn:
                        set_client_llm_allowed(conn, cname, llm_ok)
                        audit(conn, USER, "client_update", target=cname, detail={"llm_allowed": llm_ok})
                        conn.commit()
                    flash(f"顧客「{cname}」の社外 LLM を「{'使える' if llm_ok else '使わない'}」にしました。")
                    st.rerun()

                with st.expander("名前の変更・統合"):
                    st.caption("登録済みの顧客名を入力すると、その顧客に統合します（文書・用語集・担当者を付け替え、"
                               "この顧客名は消えます）。社外 LLM の設定は、どちらかが「使わない」なら「使わない」になります。")
                    rn_new = st.text_input("新しい顧客名", key=f"adm_rename_{cname}")
                    merge = rn_new.strip() in by_name and rn_new.strip() != cname
                    rn_ok = st.checkbox(
                        (f"「{cname}」を「{rn_new.strip()}」に統合します" if merge
                         else f"「{cname}」を「{rn_new.strip()}」に変更します") if rn_new.strip() else "確認",
                        key=f"adm_rename_ok_{cname}")
                    if st.button("実行", disabled=not (rn_new.strip() and rn_ok), key=f"adm_rename_do_{cname}"):
                        try:
                            with connect() as conn:
                                done = rename_client(conn, cname, rn_new)
                                audit(conn, USER, "client_rename", target=cname,
                                      detail={"to": done, "merged": merge, "documents": c["documents"],
                                              "terms": c["terms"]})
                        except ValueError as exc:
                            st.error(str(exc))
                        else:
                            flash(f"顧客「{cname}」を「{done}」に{'統合' if merge else '変更'}しました。")
                            st.rerun()

                with st.expander("この顧客を削除"):
                    if c["documents"] or c["terms"]:
                        st.caption("文書または用語集が残っているため削除できません。別の顧客に統合してください。")
                    elif st.button("削除する", key=f"adm_client_del_{cname}"):
                        try:
                            with connect() as conn:
                                delete_client(conn, cname)
                                audit(conn, USER, "client_delete", target=cname)
                        except ValueError as exc:
                            st.error(str(exc))
                        else:
                            flash(f"顧客「{cname}」を削除しました。")
                            st.rerun()

        else:  # 利用者
            if tmp := st.session_state.pop("adm_temp_password", None):
                st.success(f"{tmp[0]} の仮パスワード：**{tmp[1]}**　（この画面を閉じると再表示できません。"
                           "本人に直接伝えてください。初回ログイン時に変更を求めます）")
            try:
                with connect() as conn:
                    users = list_users(conn)
            except Exception as exc:
                users = []
                st.error(f"データベースに接続できません: {exc}")
            now_utc = pd.Timestamp.now(tz="UTC")
            st.dataframe(pd.DataFrame([{
                "ログイン ID": u["login_id"], "氏名": u["display_name"], "役割": ROLES[u["role"]],
                "担当顧客": "全顧客" if u["all_clients"] or u["role"] == "admin" else "、".join(u["clients"]),
                "状態": ("無効" if not u["active"] else
                         "ロック中" if u["locked_until"] and u["locked_until"] > now_utc else "有効"),
                "最終ログイン": format_time(u["last_login_at"]),
            } for u in users]), hide_index=True)

            with st.expander("利用者を追加"):
                with st.form("user_add", clear_on_submit=True):
                    a1, a2 = st.columns(2)
                    nu_login = a1.text_input("ログイン ID（社内メールアドレスを推奨）")
                    nu_name = a2.text_input("氏名")
                    nu_role = a1.selectbox("役割", list(ROLES), format_func=ROLES.get, index=2)
                    nu_all = a2.checkbox("全顧客を担当（管理者は常に全顧客）")
                    nu_clients = st.multiselect("担当顧客", REGISTERED_CLIENTS)
                    st.caption("仮パスワードを自動で作り、追加後に一度だけ表示します。")
                    if st.form_submit_button("追加", type="primary"):
                        pw = generate_password()
                        try:
                            with connect() as conn:
                                create_user(conn, nu_login, nu_name, nu_role, all_clients=nu_all,
                                            clients=nu_clients, password=pw)
                                audit(conn, USER, "user_create", target=nu_login.strip(), detail={
                                    "role": nu_role, "all_clients": nu_all, "clients": nu_clients})
                        except psycopg.errors.UniqueViolation:
                            st.error(f"ログイン ID「{nu_login}」は登録済みです。")
                        except ValueError as exc:
                            st.error(str(exc))
                        else:
                            st.session_state["adm_temp_password"] = (nu_login.strip(), pw)
                            flash(f"利用者「{nu_login.strip()}」を追加しました。")
                            st.rerun()

            if users:
                st.subheader("利用者の変更")
                by_id = {u["id"]: u for u in users}
                uid = st.selectbox("利用者", list(by_id), key="adm_user",
                                   format_func=lambda i: f"{by_id[i]['login_id']}　{by_id[i]['display_name']}")
                u = by_id[uid]
                with st.form(f"user_edit_{uid}"):
                    e1, e2 = st.columns(2)
                    eu_name = e1.text_input("氏名", value=u["display_name"])
                    eu_role = e2.selectbox("役割", list(ROLES), format_func=ROLES.get, index=list(ROLES).index(u["role"]))
                    eu_all = e1.checkbox("全顧客を担当", value=u["all_clients"])
                    eu_active = e2.checkbox("有効（オフにするとログインできなくなります）", value=u["active"])
                    eu_clients = st.multiselect("担当顧客", REGISTERED_CLIENTS,
                                                default=[c for c in u["clients"] if c in REGISTERED_CLIENTS])
                    if st.form_submit_button("保存", type="primary"):
                        try:
                            with connect() as conn:
                                update_user(conn, uid, display_name=eu_name, role=eu_role, all_clients=eu_all,
                                            clients=eu_clients, active=eu_active)
                                audit(conn, USER, "user_update", target=u["login_id"], detail={
                                    "role": eu_role, "all_clients": eu_all, "clients": eu_clients,
                                    "active": eu_active})
                        except ValueError as exc:
                            st.error(str(exc))
                        else:
                            flash(f"利用者「{u['login_id']}」を変更しました。")
                            st.rerun()
                p1, p2 = st.columns(2)
                if p1.button("仮パスワードを発行（再設定）", key=f"adm_pw_{uid}",
                             help="本人がパスワードを忘れた場合に使います。ロックも解除されます。"):
                    pw = generate_password()
                    with connect() as conn:
                        reset_password(conn, uid, pw)
                        audit(conn, USER, "password_reset", target=u["login_id"])
                        conn.commit()
                    st.session_state["adm_temp_password"] = (u["login_id"], pw)
                    st.rerun()
                if u["locked_until"] and u["locked_until"] > now_utc and p2.button("ロックを解除", key=f"adm_unlock_{uid}"):
                    with connect() as conn:
                        unlock_user(conn, uid)
                    flash(f"利用者「{u['login_id']}」のロックを解除しました。")
                    st.rerun()


# =====================================================================
# 操作ログ（管理者だけ）
# =====================================================================
if log_tab is not None:
    with log_tab:
        st.caption("ログイン、取り込み、削除、用語集の変更、翻訳の実行などを記録しています。"
                   "翻訳した文章そのものは記録しません。")
        try:
            with connect() as conn:
                log_users = [r[0] for r in conn.execute(
                    "SELECT DISTINCT login_id FROM audit_log WHERE login_id IS NOT NULL ORDER BY 1")]
        except Exception:
            log_users = []
        l1, l2, l3 = st.columns([2, 3, 1])
        log_user = l1.selectbox("利用者", [None] + log_users, format_func=lambda x: "すべて" if x is None else x,
                                key="log_user")
        log_actions = l2.multiselect("操作", list(ACTION_LABELS), format_func=ACTION_LABELS.get, key="log_actions")
        log_limit = l3.number_input("件数", 50, 10000, 500, step=50, key="log_limit")
        try:
            with connect() as conn:
                logs = list_audit_log(conn, limit=int(log_limit), login_id=log_user, actions=log_actions or None)
        except Exception as exc:
            logs = []
            st.error(f"データベースに接続できません: {exc}")
        log_df = pd.DataFrame([{
            "日時": format_time(r["at"]), "利用者": r["login_id"] or "", "操作": ACTION_LABELS.get(r["action"], r["action"]),
            "対象": r["target"] or "",
            "詳細": "" if r["detail"] is None else json.dumps(r["detail"], ensure_ascii=False),
        } for r in logs], columns=["日時", "利用者", "操作", "対象", "詳細"])
        st.dataframe(log_df, hide_index=True, height=480)
        st.download_button("CSV でダウンロード", data=log_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"操作ログ_{pd.Timestamp.now():%Y%m%d_%H%M}.csv", mime="text/csv",
                           key="log_download", disabled=log_df.empty)
