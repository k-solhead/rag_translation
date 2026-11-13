"""対訳表の一括投入（英文・和文の両方をベクトル化して1回で登録）。

入力ファイル
    Excel（.xlsx）: idml2xlsx の対訳表をそのまま使える
        - 「対訳表」シート（無ければ先頭シート）を読む
        - 「文書情報」シート（A列=項目、B列=値）があれば文書情報として使う
            案件名 / ファイル名 / 顧客名 / タグ / メモ
    CSV（UTF-8 / BOM 付き可）

対訳の列（いずれかの列名があればよい）
    英文      : en_text / 英語 / 英文 / source_en
    和文      : ja_text / 日本語 / 和文 / target_ja
    英ページ  : page_en / ページ(英)   … 任意。「p.12」のような表記も可
    和ページ  : page_ja / ページ(日)   … 任意
    共通ページ: page_no / ページ        … 任意（上の2列が無いときに両方へ使う）
    文書キー  : doc_key                 … 任意（旧形式。文書情報が無い場合のみ使用）
    文書名    : title                   … 任意（旧形式）

英文・和文のどちらかが空の行（片側のみ）は取り込まない。

使い方（CLI）
    python ingest.py 対訳表.xlsx
    python ingest.py 対訳表.xlsx --doc-key AGH_SR26J --title "アサヒ_サスレポ2026_和文" \\
        --client "アサヒグループホールディングス" --tags "統合報告書,FY2026" --note "..."
    python ingest.py 対訳表.xlsx --replace        # 同じ文書の既存の対訳を削除してから登録
    python ingest.py 対訳表.xlsx --defer-index    # 初回の大量投入時（HNSW を投入後に作り直す）
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import BinaryIO, Callable
from zipfile import BadZipFile

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

from common import EMBED_DIM, LANGS, Embedder, clean_text, connect, normalize_for_match
from documents import DocInfo, read_doc_info_sheet, upsert_document

TABLE_SHEET = "対訳表"
CHUNK_ROWS = 5000  # 1 回の COPY で投入する行数

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "en_text": ("en_text", "英語", "英文", "source_en"),
    "ja_text": ("ja_text", "日本語", "和文", "target_ja"),
    "page_en": ("page_en", "ページ(英)"),
    "page_ja": ("page_ja", "ページ(日)"),
    "page_no": ("page_no", "ページ"),
    "doc_key": ("doc_key",),
    "title": ("title",),
}
HNSW_INDEXES = {
    lang: (f"idx_segments_{lang}_embedding_hnsw",
           f"CREATE INDEX IF NOT EXISTS idx_segments_{lang}_embedding_hnsw ON segments "
           f"USING hnsw ({lang}_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)")
    for lang in LANGS
}
_PAGE_NUM = re.compile(r"\d+")

Progress = Callable[[float, str], None]


@dataclass
class Bitext:
    """読み込んだ対訳表（行データと、ファイル内の文書情報）。"""
    rows: list[dict]
    doc_info: DocInfo | None
    skipped: int  # 片側のみ・空行などで取り込まない行数


@dataclass
class IngestResult:
    inserted: int
    total: int
    documents: dict[str, int]  # doc_key → documents.id
    replaced: int = 0          # --replace で削除した既存の対訳数


# ---------------------------------------------------------------------
# 読み込み
# ---------------------------------------------------------------------
def _parse_page(value) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return int(value)
    m = _PAGE_NUM.search(str(value))
    return int(m.group()) if m else None


def _pick(record: Mapping, key: str):
    for name in COLUMN_ALIASES[key]:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return None


def normalize_rows(records: Iterable[Mapping]) -> tuple[list[dict], int]:
    rows: list[dict] = []
    skipped = 0
    seq_counter: dict[str | None, int] = {}
    for record in records:
        en = clean_text(str(_pick(record, "en_text") or ""))
        ja = clean_text(str(_pick(record, "ja_text") or ""))
        if not (en and ja):
            skipped += 1
            continue
        page_common = _parse_page(_pick(record, "page_no"))
        page_en = _parse_page(_pick(record, "page_en"))
        page_ja = _parse_page(_pick(record, "page_ja"))
        doc_key = _pick(record, "doc_key")
        doc_key = str(doc_key).strip() if doc_key is not None else None
        seq_counter[doc_key] = seq_counter.get(doc_key, 0) + 1
        rows.append({
            "doc_key": doc_key,
            "title": _pick(record, "title"),
            "seq_no": seq_counter[doc_key],
            "page_en": page_en if page_en is not None else page_common,
            "page_ja": page_ja if page_ja is not None else page_common,
            "en_text": en, "en_norm": normalize_for_match(en, "en"),
            "ja_text": ja, "ja_norm": normalize_for_match(ja, "ja"),
        })
    return rows, skipped


def _check_columns(headers: Iterable[str], where: str) -> None:
    headers = set(headers)
    missing = [k for k in ("en_text", "ja_text") if not headers & set(COLUMN_ALIASES[k])]
    if missing:
        names = " / ".join("・".join(COLUMN_ALIASES[k]) for k in missing)
        raise ValueError(f"{where}に英文・和文の列が見つかりません（必要な列名: {names}）")


def read_excel(source: str | BinaryIO) -> Bitext:
    """Excel から対訳表と「文書情報」シートを読む。"""
    try:
        wb = load_workbook(source, read_only=True, data_only=True)
    except (BadZipFile, InvalidFileException, OSError) as exc:
        raise ValueError("有効な .xlsx ファイルを指定してください") from exc
    try:
        ws = wb[TABLE_SHEET] if TABLE_SHEET in wb.sheetnames else wb.worksheets[0]
        values = ws.iter_rows(values_only=True)
        headers = [str(v).strip() if v is not None else "" for v in (next(values, None) or ())]
        _check_columns(headers, f"「{ws.title}」シート")
        rows, skipped = normalize_rows(dict(zip(headers, r)) for r in values)
        return Bitext(rows, read_doc_info_sheet(wb), skipped)
    finally:
        wb.close()


def read_csv(source: str | BinaryIO) -> Bitext:
    if isinstance(source, str):
        f = open(source, newline="", encoding="utf-8-sig")
    else:
        f = io.TextIOWrapper(source, encoding="utf-8-sig", newline="")
    with f:
        reader = csv.DictReader(f)
        _check_columns(reader.fieldnames or [], "CSV")
        rows, skipped = normalize_rows(reader)
    return Bitext(rows, None, skipped)


def read_bitext(path: str) -> Bitext:
    return read_csv(path) if path.lower().endswith(".csv") else read_excel(path)


# ---------------------------------------------------------------------
# ベクトル化・投入
# ---------------------------------------------------------------------
def _embed(rows: list[dict], embedder: Embedder, batch_size: int, report: Progress) -> None:
    """英文・和文それぞれ、同じ文（正規化後）は1回だけベクトル化して使い回す。"""
    for k, lang in enumerate(LANGS):
        unique: dict[str, str] = {}
        for r in rows:
            unique.setdefault(r[f"{lang}_norm"], r[f"{lang}_text"])
        keys = list(unique)
        report(0.05 + 0.4 * k, f"{'英文' if lang == 'en' else '和文'}をベクトル化中（{len(keys):,} 件）…")
        vecs = embedder.encode([unique[key] for key in keys], batch_size=batch_size)
        if vecs.shape[1] != EMBED_DIM:
            raise ValueError(f"埋め込み次元 {vecs.shape[1]} が EMBED_DIM={EMBED_DIM} と一致しません")
        lookup = dict(zip(keys, vecs))
        for r in rows:
            r[f"{lang}_embedding"] = lookup[r[f"{lang}_norm"]]


def _insert_segments(conn, rows: list[dict], doc_ids: dict[str | None, int],
                     model_name: str, allow_duplicates: bool) -> int:
    dedupe_sql = "" if allow_duplicates else """
        WHERE NOT EXISTS (
            SELECT 1 FROM segments s
             WHERE s.document_id = t.document_id
               AND s.en_hash = md5(t.en_norm)
               AND s.ja_hash = md5(t.ja_norm)
        )"""
    inserted = 0
    for start in range(0, len(rows), CHUNK_ROWS):
        chunk = rows[start:start + CHUNK_ROWS]
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TEMP TABLE IF NOT EXISTS tmp_segments (
                    document_id bigint, seq_no integer, page_ja integer, page_en integer,
                    en_text text, en_norm text, en_embedding vector({EMBED_DIM}),
                    ja_text text, ja_norm text, ja_embedding vector({EMBED_DIM})
                ) ON COMMIT DROP
            """)
            cur.execute("TRUNCATE tmp_segments")
            with cur.copy("COPY tmp_segments FROM STDIN WITH (FORMAT BINARY)") as cp:
                cp.set_types(["int8", "int4", "int4", "int4",
                              "text", "text", "vector", "text", "text", "vector"])
                for r in chunk:
                    cp.write_row([doc_ids[r["doc_key"]], r["seq_no"], r["page_ja"], r["page_en"],
                                  r["en_text"], r["en_norm"], r["en_embedding"],
                                  r["ja_text"], r["ja_norm"], r["ja_embedding"]])
            # 同じファイル内の重複（同じ英文・同じ和文）も1件にまとめる
            cur.execute(f"""
                INSERT INTO segments (document_id, seq_no, page_ja, page_en,
                                      en_text, en_norm, en_embedding,
                                      ja_text, ja_norm, ja_embedding, embed_model)
                SELECT document_id, seq_no, page_ja, page_en,
                       en_text, en_norm, en_embedding,
                       ja_text, ja_norm, ja_embedding, %s
                  FROM (SELECT DISTINCT ON (document_id, en_norm, ja_norm) *
                          FROM tmp_segments
                         ORDER BY document_id, en_norm, ja_norm, seq_no) t
                {dedupe_sql}
            """, (model_name,))
            inserted += cur.rowcount
    return inserted


def ingest_bitext(
    bitext: Bitext,
    doc_info: DocInfo | None = None,
    *,
    embedder: Embedder | None = None,
    replace: bool = False,
    batch_size: int = 64,
    defer_index: bool = False,
    allow_duplicates: bool = False,
    progress: Progress | None = None,
    dsn: str | None = None,
) -> IngestResult:
    """対訳表を登録する。

    doc_info : 文書情報。指定が無ければファイル内の「文書情報」シートを使う。
               どちらも無ければ、行ごとの doc_key 列（旧形式）で文書を分ける。
    replace  : True なら、対象文書の既存の対訳を削除してから登録する（文書情報は残す）。
    """
    report = progress or (lambda *_: None)
    rows = bitext.rows
    if not rows:
        raise ValueError("取り込める対訳がありません（英文・和文の両方が入った行が必要です）")

    info = (bitext.doc_info or DocInfo()).merged(doc_info).cleaned() if (bitext.doc_info or doc_info) else None
    if info and info.doc_key:
        for r in rows:  # 1ファイル = 1文書
            r["doc_key"] = info.doc_key
        for i, r in enumerate(rows, start=1):
            r["seq_no"] = i
    elif any(r["doc_key"] is None for r in rows):
        raise ValueError("ファイル名（doc_key）が指定されていません。登録画面または「文書情報」シートで指定してください。")

    embedder = embedder or Embedder()
    _embed(rows, embedder, batch_size, report)

    report(0.85, "データベースに登録中…")
    with connect(dsn) if dsn else connect() as conn:
        with conn.transaction():
            doc_ids: dict[str | None, int] = {}
            if info and info.doc_key:
                doc_ids[info.doc_key] = upsert_document(conn, info)
            else:  # 旧形式：行ごとの doc_key / title
                for r in rows:
                    if r["doc_key"] not in doc_ids:
                        doc_ids[r["doc_key"]] = upsert_document(
                            conn, DocInfo(doc_key=r["doc_key"], title=r["title"]))

            replaced = 0
            if replace:
                cur = conn.execute("DELETE FROM segments WHERE document_id = ANY(%s)",
                                   (list(doc_ids.values()),))
                replaced = cur.rowcount

            if defer_index:
                for name, _ in HNSW_INDEXES.values():
                    conn.execute(f"DROP INDEX IF EXISTS {name}")

            inserted = _insert_segments(conn, rows, doc_ids, embedder.model_name, allow_duplicates)

            if defer_index:
                report(0.92, "ベクトル索引を作成中…")
                conn.execute("SET LOCAL maintenance_work_mem = '1GB'")  # NAS のメモリに合わせて調整
                for _, ddl in HNSW_INDEXES.values():
                    conn.execute(ddl)
        conn.execute("ANALYZE documents")
        conn.execute("ANALYZE segments")
        conn.commit()

    report(1.0, "完了")
    return IngestResult(inserted, len(rows), {k: v for k, v in doc_ids.items() if k}, replaced)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="対訳表の一括投入（英文・和文を同時にベクトル化）")
    ap.add_argument("path", help="対訳表（.xlsx / .csv）")
    ap.add_argument("--doc-key", help="ファイル名（文書を一意に識別するキー）")
    ap.add_argument("--title", help="案件名")
    ap.add_argument("--client", help="顧客名")
    ap.add_argument("--tags", help="タグ（カンマ区切り）")
    ap.add_argument("--note", help="メモ")
    ap.add_argument("--replace", action="store_true", help="同じ文書の既存の対訳を削除してから登録")
    ap.add_argument("--batch-size", type=int, default=64, help="ベクトル化のバッチサイズ")
    ap.add_argument("--defer-index", action="store_true",
                    help="ベクトル索引を一旦削除し、投入後に作り直す（大量投入時に高速）")
    ap.add_argument("--allow-duplicates", action="store_true",
                    help="同一文書内の同じ英文・和文の組も重複登録する")
    args = ap.parse_args()

    try:
        bitext = read_bitext(args.path)
    except ValueError as exc:
        sys.exit(str(exc))
    print(f"読み込み: {len(bitext.rows):,} 行（片側のみ・空行などスキップ {bitext.skipped:,} 行）")

    override = DocInfo(args.doc_key, args.title, args.client, args.tags or [], args.note)
    t0 = time.time()
    try:
        result = ingest_bitext(
            bitext, override, replace=args.replace, batch_size=args.batch_size,
            defer_index=args.defer_index, allow_duplicates=args.allow_duplicates,
            progress=lambda frac, msg: print(f"[{frac:4.0%}] {msg}"),
        )
    except ValueError as exc:
        sys.exit(str(exc))
    if result.replaced:
        print(f"既存の対訳を削除: {result.replaced:,} 行")
    print(f"登録: 新規 {result.inserted:,} 行 / 対象 {result.total:,} 行"
          f"（文書: {', '.join(result.documents)}）/ {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()
