"""documents テーブル（文書情報）の読み書き。

1つの対訳表ファイル = 1文書。画面の登録欄・対訳表の「文書情報」シート・CLI 引数の
いずれから来た情報も DocInfo にまとめて扱う。

    画面の項目   列名      例
    ファイル名   doc_key   AGH_SR26J（一意）
    案件名       title     アサヒ_サスレポ2026_和文
    顧客名       client    アサヒグループホールディングス
    タグ         tags      統合報告書, FY2026
    メモ         note      自由記述
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# 「文書情報」シートの項目名 → DocInfo の属性名
DOC_INFO_SHEET = "文書情報"
DOC_INFO_LABELS = {
    "案件名": "title",
    "ファイル名": "doc_key",
    "顧客名": "client",
    "タグ": "tags",
    "メモ": "note",
}

_TAG_SPLIT = re.compile(r"[,、，;；\n]+")


def clean_field(value) -> str | None:
    """前後の空白除去と全角半角の統一（NFKC）。空なら None。"""
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    return text or None


def parse_tags(value) -> list[str]:
    """「統合報告書, FY2026、IR」のような文字列（またはリスト）をタグのリストにする。"""
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else _TAG_SPLIT.split(str(value))
    tags: list[str] = []
    for item in items:
        tag = clean_field(item)
        if tag and tag not in tags:
            tags.append(tag)
    return tags


@dataclass
class DocInfo:
    doc_key: str | None = None
    title: str | None = None
    client: str | None = None
    tags: list[str] = field(default_factory=list)
    note: str | None = None

    def cleaned(self) -> "DocInfo":
        note = (self.note or "").strip() or None  # メモは改行を保つため NFKC しない
        return DocInfo(clean_field(self.doc_key), clean_field(self.title),
                       clean_field(self.client), parse_tags(self.tags), note)

    def merged(self, override: "DocInfo | None") -> "DocInfo":
        """override 側に値がある項目だけ上書きした DocInfo を返す。"""
        if override is None:
            return self
        return DocInfo(
            override.doc_key or self.doc_key,
            override.title or self.title,
            override.client or self.client,
            override.tags or self.tags,
            override.note or self.note,
        )


def read_doc_info_sheet(workbook) -> DocInfo | None:
    """openpyxl のブックから「文書情報」シート（A列=項目、B列=値）を読む。無ければ None。"""
    if DOC_INFO_SHEET not in workbook.sheetnames:
        return None
    info = DocInfo()
    for row in workbook[DOC_INFO_SHEET].iter_rows(values_only=True):
        if not row or row[0] is None:
            continue
        attr = DOC_INFO_LABELS.get(str(row[0]).strip())
        value = row[1] if len(row) > 1 else None
        if attr == "tags":
            info.tags = parse_tags(value)
        elif attr:
            setattr(info, attr, value)
    return info.cleaned()


# ---------------------------------------------------------------------
# DB 操作
# ---------------------------------------------------------------------
def upsert_document(conn, info: DocInfo) -> int:
    """doc_key で登録または更新し、documents.id を返す。

    既存文書の場合、値が入っている項目だけを更新する（空欄で既存の値を消さない）。
    値を消したい場合は update_document() を使う（画面の「文書管理」タブ）。
    """
    info = info.cleaned()
    if not info.doc_key:
        raise ValueError("ファイル名（doc_key）は必須です")
    row = conn.execute(
        """
        INSERT INTO documents (doc_key, title, client, tags, note)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (doc_key) DO UPDATE SET
            title  = COALESCE(EXCLUDED.title,  documents.title),
            client = COALESCE(EXCLUDED.client, documents.client),
            tags   = CASE WHEN cardinality(EXCLUDED.tags) > 0
                          THEN EXCLUDED.tags ELSE documents.tags END,
            note   = COALESCE(EXCLUDED.note,   documents.note)
        RETURNING id
        """,
        (info.doc_key, info.title, info.client, info.tags, info.note),
    ).fetchone()
    return row[0]


def get_document(conn, doc_key: str) -> dict | None:
    cur = conn.execute(
        """
        SELECT d.id, d.doc_key, d.title, d.client, d.tags, d.note,
               (SELECT count(*) FROM segments s WHERE s.document_id = d.id) AS segments
          FROM documents d WHERE d.doc_key = %s
        """,
        (clean_field(doc_key),),
    )
    row = cur.fetchone()
    return dict(zip([c.name for c in cur.description], row)) if row else None


def list_documents(conn) -> list[dict]:
    cur = conn.execute(
        """
        SELECT d.id, d.doc_key, d.title, d.client, d.tags, d.note,
               count(s.id) AS segments, d.created_at, d.updated_at
          FROM documents d
          LEFT JOIN segments s ON s.document_id = d.id
         GROUP BY d.id
         ORDER BY d.created_at DESC, d.id DESC
        """
    )
    cols = [c.name for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def update_document(conn, doc_id: int, info: DocInfo) -> None:
    """文書情報をそのまま上書きする（空欄は空にする）。"""
    info = info.cleaned()
    if not info.doc_key:
        raise ValueError("ファイル名（doc_key）は必須です")
    conn.execute(
        "UPDATE documents SET doc_key=%s, title=%s, client=%s, tags=%s, note=%s WHERE id=%s",
        (info.doc_key, info.title, info.client, info.tags, info.note, doc_id),
    )


def delete_document(conn, doc_id: int) -> None:
    """文書と、その文書の対訳（segments）をすべて削除する。"""
    conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))


def list_clients(conn) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT client FROM documents WHERE client IS NOT NULL ORDER BY client")]


def list_tags(conn) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT unnest(tags) AS tag FROM documents ORDER BY tag")]


def find_document_ids(conn, *, clients: list[str] | None = None,
                      tags: list[str] | None = None, tag_mode: str = "all") -> list[int]:
    """顧客名（いずれか）とタグで文書を絞り込み、id のリストを返す。

    tag_mode="all": 指定タグをすべて持つ文書 / "any": いずれかを持つ文書
    """
    where, params = [], []
    if clients:
        where.append("client = ANY(%s)")
        params.append(list(clients))
    if tags:
        where.append("tags @> %s" if tag_mode == "all" else "tags && %s")
        params.append(parse_tags(tags))
    sql = "SELECT id FROM documents" + (" WHERE " + " AND ".join(where) if where else "")
    return [r[0] for r in conn.execute(sql, params)]
