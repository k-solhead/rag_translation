"""過去翻訳データの一括投入スクリプト（英文のベクトル化を含む）。
 
入力 CSV（UTF-8 / BOM 付き可）または Excel（先頭シート）の列:
    doc_key    必須  文書を一意に識別するキー（案件番号・ファイル名など）
    title      任意  文書名
    page_no    任意  ページ番号（整数）
    source_en  必須  英文
    target_ja  必須  和訳
 
使い方:
    python ingest.py data.csv
    python ingest.py data.csv --defer-index      # 初回の大量投入時（HNSW を投入後に作り直す）
    python ingest.py data.csv --allow-duplicates # 同一文書内の同一英文・同一訳も重複登録する
 
処理の流れ:
    1. CSV を読み込み、正規化・検証
    2. documents を UPSERT して doc_key → id を取得
    3. 英文を重複排除してからベクトル化（同じ文が多いマニュアル類で大幅に時短）
    4. 一時テーブルへ COPY（バイナリ）→ 重複を除いて segments へ INSERT
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Iterable, Mapping
from collections import defaultdict
from typing import BinaryIO
from zipfile import BadZipFile

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

from common import EMBED_DIM, Embedder, clean_text, connect, normalize_for_match
 
REQUIRED_COLS = {"doc_key", "source_en", "target_ja"}
CHUNK_ROWS = 5000  # 1 トランザクションで投入する行数
 
def normalize_rows(records: Iterable[Mapping[str, object]]) -> list[dict]:
    rows: list[dict] = []
    skipped = 0

    seq_counter: dict[str, int] = defaultdict(int)
    for line_no, record in enumerate(records, start=2):
        def cell_text(name: str) -> str:
            value = record.get(name)
            return "" if value is None else str(value).strip()

        src = clean_text(cell_text("source_en"))
        tgt = cell_text("target_ja")
        doc_key = cell_text("doc_key")
        if not (src and tgt and doc_key):
            skipped += 1
            continue

        page_value = record.get("page_no")
        page_raw = cell_text("page_no")
        try:
            if isinstance(page_value, float) and page_value.is_integer():
                page_no = int(page_value)
            else:
                page_no = int(page_raw) if page_raw else None
        except ValueError:
            print(f"[warn] {line_no}行目: page_no が整数ではないため空にします: {page_raw!r}")
            page_no = None

        seq_counter[doc_key] += 1
        rows.append({
            "doc_key": doc_key,
            "title": cell_text("title") or None,
            "page_no": page_no,
            "seq_no": seq_counter[doc_key],
            "source_en": src,
            "source_norm": normalize_for_match(src),
            "target_ja": tgt,
        })
    print(f"読み込み: {len(rows):,} 行（空行などスキップ {skipped:,} 行）")
    return rows


def read_rows(path: str) -> list[dict]:
    """CSV ファイルから投入対象の行を読み込む。"""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLS - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"CSV に必須列がありません: {sorted(missing)}")
        return normalize_rows(reader)


def read_excel_rows(source: BinaryIO) -> list[dict]:
    """Excel ファイルの先頭シートから投入対象の行を読み込む。"""
    try:
        workbook = load_workbook(source, read_only=True, data_only=True)
    except (BadZipFile, InvalidFileException, OSError) as exc:
        raise ValueError("有効な .xlsx ファイルを指定してください") from exc

    try:
        worksheet = workbook.worksheets[0]
        values = worksheet.iter_rows(values_only=True)
        header_row = next(values, None)
        headers = [
            str(value).strip() if value is not None else ""
            for value in (header_row or ())
        ]
        missing = REQUIRED_COLS - set(headers)
        if missing:
            raise ValueError(f"Excel の先頭シートに必須列がありません: {sorted(missing)}")

        records = (
            dict(zip(headers, row))
            for row in values
        )
        return normalize_rows(records)
    finally:
        workbook.close()


#  文書管理テーブルに新規文書を追加
def upsert_documents(conn, rows: list[dict]) -> dict[str, int]:
    docs: dict[str, str | None] = {}
    for r in rows:
        docs.setdefault(r["doc_key"], r["title"])
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO documents (doc_key, title) VALUES (%s, %s)
            ON CONFLICT (doc_key) DO UPDATE
               SET title = COALESCE(EXCLUDED.title, documents.title)
            """,
            list(docs.items()),
        )
        cur.execute("SELECT doc_key, id FROM documents WHERE doc_key = ANY(%s)", (list(docs),))
        mapping = dict(cur.fetchall())
    conn.commit()
    print(f"文書: {len(mapping):,} 件")
    return mapping
 
 
def embed_rows(rows: list[dict], embedder: Embedder, batch_size: int) -> None:
    """同じ英文（正規化後）は1回だけベクトル化して使い回す。"""
    unique: dict[str, str] = {}
    for r in rows:
        unique.setdefault(r["source_norm"], r["source_en"])
    keys = list(unique)
    print(f"ベクトル化: ユニーク英文 {len(keys):,} 件（全 {len(rows):,} 行）")
 
    t0 = time.time()
    vecs = embedder.encode([unique[k] for k in keys], batch_size=batch_size, show_progress=True)
    if vecs.shape[1] != EMBED_DIM:
        sys.exit(f"埋め込み次元 {vecs.shape[1]} が EMBED_DIM={EMBED_DIM} と一致しません")
    lookup = dict(zip(keys, vecs))
    for r in rows:
        r["embedding"] = lookup[r["source_norm"]]
    print(f"ベクトル化完了: {time.time() - t0:.1f} 秒")
 
 
def insert_segments(conn, rows: list[dict], doc_ids: dict[str, int],
                    model_name: str, allow_duplicates: bool) -> int:
    dedupe_sql = "" if allow_duplicates else """
        WHERE NOT EXISTS (
            SELECT 1 FROM segments s
             WHERE s.document_id = t.document_id
               AND s.source_hash = md5(t.source_norm)
               AND s.target_ja   = t.target_ja
        )"""
    inserted = 0
    for start in range(0, len(rows), CHUNK_ROWS):
        chunk = rows[start:start + CHUNK_ROWS]
        with conn.transaction(), conn.cursor() as cur:
            cur.execute("""
                CREATE TEMP TABLE tmp_segments (
                    document_id bigint, page_no integer, seq_no integer,
                    source_en text, source_norm text, target_ja text,
                    embedding vector(%s)
                ) ON COMMIT DROP
            """ % EMBED_DIM)
            with cur.copy("COPY tmp_segments FROM STDIN WITH (FORMAT BINARY)") as cp:
                cp.set_types(["int8", "int4", "int4", "text", "text", "text", "vector"])
                for r in chunk:
                    cp.write_row([doc_ids[r["doc_key"]], r["page_no"], r["seq_no"],
                                  r["source_en"], r["source_norm"], r["target_ja"],
                                  r["embedding"]])
            # 同じ CSV 内での重複も除外する（DISTINCT ON）
            cur.execute(f"""
                INSERT INTO segments (document_id, page_no, seq_no, source_en, source_norm,
                                      target_ja, embedding, embed_model)
                SELECT document_id, page_no, seq_no, source_en, source_norm,
                       target_ja, embedding, %s
                  FROM (SELECT DISTINCT ON (document_id, source_norm, target_ja) *
                          FROM tmp_segments
                         ORDER BY document_id, source_norm, target_ja, seq_no) t
                {dedupe_sql}
            """, (model_name,))
            inserted += cur.rowcount
        print(f"  投入 {min(start + CHUNK_ROWS, len(rows)):,}/{len(rows):,} 行（新規 {inserted:,}）")
    return inserted
 
 
def ingest_rows(
    rows: list[dict],
    batch_size: int = 64,
    defer_index: bool = False,
    allow_duplicates: bool = False,
) -> int:
    if not rows:
        raise ValueError("投入対象がありません")

    embedder = Embedder()
    embed_rows(rows, embedder, batch_size)

    with connect() as conn:
        doc_ids = upsert_documents(conn, rows)

        if defer_index:
            conn.execute("DROP INDEX IF EXISTS idx_segments_embedding_hnsw")
            conn.commit()

        t0 = time.time()
        n = insert_segments(conn, rows, doc_ids, embedder.model_name, allow_duplicates)
        print(f"segments 投入完了: 新規 {n:,} 行 / {time.time() - t0:.1f} 秒")

        if defer_index:
            print("HNSW インデックスを作成中…")
            t0 = time.time()
            conn.execute("SET maintenance_work_mem = '1GB'")  # NAS のメモリに合わせて調整
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_segments_embedding_hnsw
                    ON segments USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
            """)
            conn.commit()
            print(f"インデックス作成完了: {time.time() - t0:.1f} 秒")
 
        conn.execute("ANALYZE documents")
        conn.execute("ANALYZE segments")
        conn.commit()
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="過去翻訳データ一括投入")
    ap.add_argument("csv_path")
    ap.add_argument("--batch-size", type=int, default=64, help="ベクトル化のバッチサイズ")
    ap.add_argument("--defer-index", action="store_true",
                    help="HNSW インデックスを一旦削除し、投入後に作り直す（大量投入時に高速）")
    ap.add_argument("--allow-duplicates", action="store_true",
                    help="同一文書内の同一英文・同一訳の重複登録を許可する")
    args = ap.parse_args()

    rows = read_rows(args.csv_path)
    if not rows:
        sys.exit("投入対象がありません")
    ingest_rows(rows, args.batch_size, args.defer_index, args.allow_duplicates)
 
 
if __name__ == "__main__":
    main()
 