-- =====================================================================
-- 過去翻訳検索（翻訳メモリ）用スキーマ
--   PostgreSQL 16 以降 / pgvector 0.7 以降 / pg_trgm
--   埋め込みモデル: intfloat/multilingual-e5-small（384次元）を想定
--   ※モデルを変える場合は vector(384) の次元数を合わせて変更すること
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------------------------------------------------------------------
-- 文書（案件・ファイル）単位の情報
--   ページ番号は「どの文書の何ページか」が分かって初めて意味を持つため、
--   セグメントとは別テーブルで管理する
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id             bigserial    PRIMARY KEY,
    doc_key        text         NOT NULL UNIQUE,   -- 案件番号・ファイル名など一意なキー
    title          text,                           -- 文書名
    client         text,                           -- 顧客名・部署名など（任意）
    translated_at  date,                           -- 翻訳実施日（任意）
    note           text,
    created_at     timestamptz  NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------
-- セグメント（英文1文 ↔ 和訳1文）
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS segments (
    id           bigserial    PRIMARY KEY,
    document_id  bigint       NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_no      integer,                          -- ページ番号
    seq_no       integer,                          -- 文書内の出現順（差分表示・前後文脈の取得用）
    source_en    text         NOT NULL,            -- 英文（原文そのまま）
    source_norm  text         NOT NULL,            -- 検索用に正規化した英文（小文字化・空白統一など）
    source_hash  text         GENERATED ALWAYS AS (md5(source_norm)) STORED,  -- 完全一致の高速判定用
    target_ja    text         NOT NULL,            -- 翻訳和文
    embedding    vector(384),                      -- pgvector 用ベクトル
    embed_model  text,                             -- ベクトル化に使ったモデル名（モデル変更時の再計算判定用）
    is_active    boolean      NOT NULL DEFAULT true,  -- 誤訳・旧訳を論理削除するためのフラグ
    created_at   timestamptz  NOT NULL DEFAULT now(),
    updated_at   timestamptz  NOT NULL DEFAULT now()
);

-- ベクトル検索（コサイン距離）用 HNSW インデックス
--   大量の初回投入時は、投入後に作成したほうが速い（ingest.py の --defer-index）
CREATE INDEX IF NOT EXISTS idx_segments_embedding_hnsw
    ON segments USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- トライグラム類似検索（% 演算子 / similarity()）用 GIN インデックス
CREATE INDEX IF NOT EXISTS idx_segments_source_trgm
    ON segments USING gin (source_norm gin_trgm_ops);

-- 完全一致・重複判定用
CREATE INDEX IF NOT EXISTS idx_segments_source_hash
    ON segments (source_hash);

-- 文書・ページ単位での参照用
CREATE INDEX IF NOT EXISTS idx_segments_doc_page
    ON segments (document_id, page_no, seq_no);

-- updated_at 自動更新
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_segments_updated_at ON segments;
CREATE TRIGGER trg_segments_updated_at
    BEFORE UPDATE ON segments
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
