-- =====================================================================
-- 過去翻訳検索（翻訳メモリ）用スキーマ ― 日英両方向対応版
--   PostgreSQL 16 以降 / pgvector / pg_trgm
--   埋め込みモデル: intfloat/multilingual-e5-small（384次元・日英両対応）
--   ※モデルを変える場合は vector(384) の次元数を合わせて変更すること
--   ※pg_trgm で日本語を扱うため、DB は UTF-8 ロケールで作成すること
--     （公式 Docker イメージの既定 en_US.utf8 で可。C ロケールは不可）
--
--   1行 = 1組の対訳（英文と和文）。英文・和文それぞれに
--   正規化文・ハッシュ・ベクトルを持たせ、英→和・和→英のどちらの検索にも使う。
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- updated_at 自動更新用の共通関数
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ---------------------------------------------------------------------
-- 文書（1つの対訳表ファイル = 1文書）
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id             bigserial    PRIMARY KEY,
    doc_key        text         NOT NULL UNIQUE,         -- ファイル名（例: AGH_SR26J）。一意
    title          text,                                 -- 案件名（例: アサヒ_サスレポ2026_和文）
    client         text,                                 -- 顧客名（例: アサヒグループホールディングス）
    tags           text[]       NOT NULL DEFAULT '{}',   -- タグ（例: {統合報告書,FY2026}）
    note           text,                                 -- メモ（自由記述）
    translated_at  date,                                 -- 翻訳実施日（任意）
    created_at     timestamptz  NOT NULL DEFAULT now(),
    updated_at     timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_client ON documents (client);
CREATE INDEX IF NOT EXISTS idx_documents_tags   ON documents USING gin (tags);

DROP TRIGGER IF EXISTS trg_documents_updated_at ON documents;
CREATE TRIGGER trg_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------
-- セグメント（英文1文 ↔ 和文1文。1対1にならない箇所は最大2文をまとめて1行）
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS segments (
    id            bigserial    PRIMARY KEY,
    document_id   bigint       NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    seq_no        integer,                       -- 文書内の出現順（前後の文脈の取得用）
    page_ja       integer,                       -- 和文版のページ番号
    page_en       integer,                       -- 英文版のページ番号

    -- 英文
    en_text       text         NOT NULL,         -- 原文そのまま
    en_norm       text         NOT NULL,         -- 検索用に正規化（小文字化・空白統一など）
    en_hash       text         GENERATED ALWAYS AS (md5(en_norm)) STORED,
    en_embedding  vector(384),

    -- 和文
    ja_text       text         NOT NULL,
    ja_norm       text         NOT NULL,         -- 検索用に正規化（全角半角統一・空白除去など）
    ja_hash       text         GENERATED ALWAYS AS (md5(ja_norm)) STORED,
    ja_embedding  vector(384),

    embed_model   text,                          -- ベクトル化に使ったモデル名
    is_active     boolean      NOT NULL DEFAULT true,  -- 誤訳・旧訳の論理削除用
    created_at    timestamptz  NOT NULL DEFAULT now(),
    updated_at    timestamptz  NOT NULL DEFAULT now()
);

-- ベクトル検索（コサイン距離）用 HNSW：英文・和文それぞれ
CREATE INDEX IF NOT EXISTS idx_segments_en_embedding_hnsw
    ON segments USING hnsw (en_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_segments_ja_embedding_hnsw
    ON segments USING hnsw (ja_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- トライグラム類似検索用 GIN：英文・和文それぞれ
CREATE INDEX IF NOT EXISTS idx_segments_en_trgm ON segments USING gin (en_norm gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_segments_ja_trgm ON segments USING gin (ja_norm gin_trgm_ops);

-- 完全一致・重複判定用
CREATE INDEX IF NOT EXISTS idx_segments_en_hash ON segments (en_hash);
CREATE INDEX IF NOT EXISTS idx_segments_ja_hash ON segments (ja_hash);

-- 文書単位の絞り込み・前後文脈の取得用
CREATE INDEX IF NOT EXISTS idx_segments_doc_seq ON segments (document_id, seq_no);

DROP TRIGGER IF EXISTS trg_segments_updated_at ON segments;
CREATE TRIGGER trg_segments_updated_at
    BEFORE UPDATE ON segments
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------
-- 用語集（共通 → 会社 → 媒体の 3 段階。client = '' は全顧客共通、medium = '' は会社全体）
--   ※既存の DB には、アプリ起動時に自動で作成される（glossary.py の ensure_glossary_table）
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS glossary_terms (
    id          bigserial    PRIMARY KEY,
    client      text         NOT NULL DEFAULT '',   -- 顧客名（'' = 共通）
    medium      text         NOT NULL DEFAULT '',   -- 媒体名（'' = 会社全体。例：統合報告書）
    en          text         NOT NULL,              -- 英語の用語
    ja          text         NOT NULL,              -- 日本語の訳語
    note        text,                               -- 備考
    required    boolean      NOT NULL DEFAULT true, -- true=必須（毎回この訳語）/ false=参考
    created_at  timestamptz  NOT NULL DEFAULT now(),
    updated_at  timestamptz  NOT NULL DEFAULT now(),
    CONSTRAINT glossary_terms_client_medium_en_ja_key UNIQUE (client, medium, en, ja)
);
CREATE INDEX IF NOT EXISTS idx_glossary_terms_client_medium ON glossary_terms (client, medium);

DROP TRIGGER IF EXISTS trg_glossary_terms_updated_at ON glossary_terms;
CREATE TRIGGER trg_glossary_terms_updated_at
    BEFORE UPDATE ON glossary_terms
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------
-- クラウド LLM のトークン使用量の累計（プロバイダーごと。全員分を合算）
--   ※既存の DB には、アプリ起動時に自動で作成される（usage.py の ensure_usage_table）
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS token_usage (
    provider            text         PRIMARY KEY,           -- anthropic / openai / gemini
    input_tokens        bigint       NOT NULL DEFAULT 0,    -- 入力（キャッシュ以外）
    output_tokens       bigint       NOT NULL DEFAULT 0,    -- 出力（推論分を含む）
    cache_read_tokens   bigint       NOT NULL DEFAULT 0,    -- キャッシュ入力（読み込み）
    cache_write_tokens  bigint       NOT NULL DEFAULT 0,    -- キャッシュ出力（書き込み）
    requests            bigint       NOT NULL DEFAULT 0,    -- 呼び出し回数
    updated_at          timestamptz,                        -- 最後に加算した日時
    reset_at            timestamptz  NOT NULL DEFAULT now() -- 集計開始（最後にゼロに戻した）日時
);
