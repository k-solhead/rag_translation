# rag-translation を develop-auth1 に更新する手順書

master ブランチの内容で動いている `rag-translation` コンテナ（DB: `tm-pg` / アプリ: `tm-app`）を、ユーザー認証とデータへのアクセス権限を追加した `develop-auth1` ブランチの内容に更新します。既存のデータベース（`tm-pg` のボリューム）はそのまま使い、作り直しません。

---

## 0. 変更の概要と前提

### 0-1. DB の変更は「追加のみ」

`develop-auth1` で DB に加わるのは次の 4 テーブルだけです。既存テーブル（`documents` / `segments` / `glossary_terms` / `token_usage`）の列や中身は変わりません。

| 新規テーブル | 役割 |
|---|---|
| `clients` | 顧客マスター（既存の文書・用語集の顧客名は起動時に自動で登録されます） |
| `app_users` | 利用者（役割：管理者・編集者・翻訳者） |
| `user_clients` | 利用者の担当顧客 |
| `audit_log` | 操作ログ |

- 新しいテーブルは、アプリの起動時に `auth.py` の `ensure_auth_tables()` が自動で作ります。
- `schema.sql` は、データベースが空のときに 1 回だけ実行される仕組みです（`Dockerfile.db` の `docker-entrypoint-initdb.d`）。既存のボリュームがあるため、今回は再実行されません。そのため `schema.sql` の変更は既存 DB には効きません。認証テーブルはアプリ側で作られるので、これで問題ありません。

### 0-2. 絶対にしてはいけないこと

- **`docker compose down -v` は実行しない。** `-v` を付けると DB のボリューム（`tm-pgdata`）が削除され、全データが失われます。
- `tm-pgdata` ボリュームを削除・再作成しない。

### 0-3. 作業前に確認が必要なこと

- **作業ツリーの `llm.py` に未コミットの変更があります。** 現在の作業ツリーでは、ローカル LLM（LM Studio / Ollama）の接続処理が有効になっています。一方、`develop-auth1` のコミット版ではこの処理はコメントアウトされています。ビルドは作業ツリーのファイルから行われるため、何もしないとこの変更がそのままイメージに入ります。手順 3 で扱いを決めてください。
- **初期管理者の設定が必要です。** 利用者が 1 人もいない状態でアプリを起動すると、`.env` の `INITIAL_ADMIN_LOGIN` と `INITIAL_ADMIN_PASSWORD` から管理者が作られます。これが未設定のままだと、誰もログインできなくなります。手順 4 で必ず設定してください。
- パスワードや API キーは、チャットや Git に貼り付けないでください。`.env` は Git の管理対象外です。

### 0-4. 想定する環境

- リポジトリ: `/Users/solhead/AI/work/rag_translation`
- Docker Compose プロジェクト名: `rag-translation`（`compose.yml` の `name:` で指定）
- 名前付きボリューム: `rag-translation_tm-pgdata`（手順 1 で実際の名前を確認します）

以下のコマンドは、コンテナが動いている Docker 環境（このリポジトリがあるマシン）で実行してください。QNAP など別の機器で動かしている場合は、`git` の操作をその機器のリポジトリ上で行ってください。

---

## 1. 現状の確認（変更前）

```bash
cd /Users/solhead/AI/work/rag_translation

# コンテナの状態（tm-pg / tm-app が running・healthy であること）
docker compose ps

# データボリューム名の確認（rag-translation_tm-pgdata が表示されること）
docker volume ls | grep tm-pgdata

# 作業ツリーの状態と現在のコミット
git status
git log --oneline -1
```

変更前の件数を記録します。移行後の件数と比べて、データが欠けていないことを確認します。

```bash
docker compose exec db psql -U tm -d tm -c "
SELECT (SELECT count(*) FROM documents)      AS documents,
       (SELECT count(*) FROM segments)       AS segments,
       (SELECT count(*) FROM glossary_terms) AS glossary_terms,
       (SELECT count(*) FROM token_usage)    AS token_usage;"
```

結果の数字は、メモ帳などに控えておいてください。

---

## 2. バックアップの取得

### 2-1. 論理バックアップ（停止せずに取得）

`pg_dump` を DB コンテナの中で実行するため、PostgreSQL のバージョンが一致します。

```bash
cd /Users/solhead/AI/work/rag_translation
BACKUP_DIR="$HOME/rag_backup/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# カスタム形式（復元に使う。pg_restore で読む）
docker compose exec -T db pg_dump -U tm -d tm -Fc > "$BACKUP_DIR/tm_full.dump"

# SQL 形式（テキストとして内容を確認できる）
docker compose exec -T db pg_dump -U tm -d tm > "$BACKUP_DIR/tm_full.sql"

# 設定と現在の状態の記録
cp .env "$BACKUP_DIR/env.backup"
git rev-parse HEAD > "$BACKUP_DIR/git_head_before.txt"
git diff > "$BACKUP_DIR/working_tree_before.patch"

ls -lh "$BACKUP_DIR"
```

- `env.backup` には秘密情報（パスワードや API キー）が含まれます。保管場所に注意してください。
- `tm_full.dump` が 0 バイトでないことを確認してください。

カスタム形式のダンプが読めるかを確認します。

```bash
docker compose exec -T db pg_restore --list < "$BACKUP_DIR/tm_full.dump" | head -20
```

テーブル名の一覧（`documents` / `segments` など）が表示されれば成功です。

### 2-2. 復元テスト（別コンテナで実施。本番には触れません）

バックアップが本当に復元できるかを、一時的な別コンテナで確かめます。

```bash
docker run -d --name tm-restore-test -e POSTGRES_PASSWORD=test pgvector/pgvector:pg16
until docker exec tm-restore-test pg_isready -U postgres; do sleep 2; done

docker exec tm-restore-test psql -U postgres -c "CREATE DATABASE tm;"
docker cp "$BACKUP_DIR/tm_full.dump" tm-restore-test:/tmp/tm.dump
docker exec tm-restore-test pg_restore -U postgres -d tm --no-owner /tmp/tm.dump

docker exec tm-restore-test psql -U postgres -d tm -c "
SELECT (SELECT count(*) FROM documents)      AS documents,
       (SELECT count(*) FROM segments)       AS segments,
       (SELECT count(*) FROM glossary_terms) AS glossary_terms,
       (SELECT count(*) FROM token_usage)    AS token_usage;"

docker rm -f tm-restore-test
```

手順 1 で控えた件数と一致すれば、バックアップは正しく復元できます。

### 2-3. ボリューム全体のバックアップ（停止して取得）

DB の内容だけでなく、データディレクトリのファイルそのものも保存します。ここでは一時的に停止します。

```bash
cd /Users/solhead/AI/work/rag_translation
docker compose stop app db

docker run --rm \
  -v rag-translation_tm-pgdata:/data:ro \
  -v "$BACKUP_DIR":/backup \
  alpine tar czf /backup/tm-pgdata_volume.tar.gz -C /data .

ls -lh "$BACKUP_DIR"
docker compose start db app
docker compose ps
```

- 停止中に `tm-pgdata_volume.tar.gz` が作られていることを確認してから `start` してください。
- `start` 後、`tm-pg` が healthy になるまで数十秒待ちます。

> バックアップは `$BACKUP_DIR` にまとめて保管してください。移行が完了し、動作を確認できるまでは削除しないでください。

---

## 3. コードを develop-auth1 に切り替える

### 3-1. 作業ツリーの `llm.py` の変更を退避する

`develop-auth1` の内容でデプロイする場合は、次のように退避します（手順 2-1 の `working_tree_before.patch` にも内容が残っています）。

```bash
cd /Users/solhead/AI/work/rag_translation
git stash push -m "llm.py local (LM Studio/Ollama enabled) before develop-auth1" -- llm.py
git stash list
```

> **判断が必要な場合**：ローカル LLM（LM Studio / Ollama）を今後も使い続けるなら、退避せずに残してください。その場合は、`develop-auth1` の `llm.py` からの差分であることを理解したうえで、ビルドに含まれることを確認してください。

### 3-2. develop-auth1 に切り替えて最新化する

```bash
cd /Users/solhead/AI/work/rag_translation
git status                       # 変更が無いことを確認（llm.py を退避した場合）
git fetch origin
git switch develop-auth1
git pull --ff-only origin develop-auth1
git log --oneline -3             # cd7b48f（major update of login process）が先頭であること
git status                       # "working tree clean" であること
```

`.env` と `.venv` は Git の管理対象外のため、切り替えの影響を受けません。

---

## 4. `.env` の更新

`.env` を編集します（エディタで開いて、直接入力してください）。

```bash
open -e .env     # または任意のエディタで開く
```

次の項目を追加します。値はご自身で入力してください。

```dotenv
# ---- ログイン ----
AUTH_MODE=local
# 初期管理者（利用者が 1 人もいないときだけ作られる）
INITIAL_ADMIN_LOGIN=（管理者の社内メールアドレス）
INITIAL_ADMIN_PASSWORD=（10文字以上。英大文字・英小文字・数字のうち 2 種類以上）
INITIAL_ADMIN_NAME=管理者
SESSION_TIMEOUT_MINUTES=60
AUTH_MAX_FAILED=5
AUTH_LOCK_MINUTES=15
# 0.0.0.0 は従来どおり LAN に公開。HTTPS のリバースプロキシ経由のみにする場合は 127.0.0.1
APP_BIND=0.0.0.0
```

- `POSTGRES_PASSWORD` など既存の項目は変更しません。
- `INITIAL_ADMIN_*` は、利用者がいない初回起動時にだけ使われます。管理者でログインできたら `INITIAL_ADMIN_PASSWORD` は空にしてかまいません。

設定ファイルの構文を確認します（値の内容は表示されません）。

```bash
docker compose config -q && echo "compose 設定 OK"
```

---

## 5. ビルドと起動

```bash
cd /Users/solhead/AI/work/rag_translation
docker compose build
docker compose up -d
docker compose ps
```

- `app` のビルドには数分〜十数分かかります（PyTorch と埋め込みモデルの同梱）。
- `schema.sql` が変わったため `db` も再ビルドされ、`tm-pg` が作り直されます。データはボリュームに残るため、数十秒の停止で済みます。初期化スクリプトは既存データがあるため再実行されません。
- `tm-pg` が `healthy`、`tm-app` が `running` になるまで待ちます。

起動ログを確認します（終了は Ctrl+C）。

```bash
docker compose logs -f app
```

エラーが出ていないこと（特に `ensure_schema` や DB 接続に関するもの）を確認してください。

---

## 6. 移行後の確認

### 6-1. テーブルとデータ件数

> 新しいテーブルは、アプリの画面に最初にアクセスしたときに作られます。コンテナを起動しただけでは作られないため、`\dt` の確認の前に、先に 6-3 の 1 を実行してブラウザで `http://localhost:8501` を開いてください。

```bash
docker compose exec db psql -U tm -d tm -c "\dt"
```

`clients` / `app_users` / `user_clients` / `audit_log` が増えていれば、テーブルは作成されています。

手順 1 で控えた件数と同じであることを確認します。

```bash
docker compose exec db psql -U tm -d tm -c "
SELECT (SELECT count(*) FROM documents)      AS documents,
       (SELECT count(*) FROM segments)       AS segments,
       (SELECT count(*) FROM glossary_terms) AS glossary_terms,
       (SELECT count(*) FROM token_usage)    AS token_usage;"
```

### 6-2. 自動で作られた顧客・管理者・操作ログ

```bash
docker compose exec db psql -U tm -d tm -c "SELECT name, llm_allowed FROM clients ORDER BY name;"
docker compose exec db psql -U tm -d tm -c "SELECT login_id, role, must_change_password FROM app_users;"
docker compose exec db psql -U tm -d tm -c "SELECT at, action, target FROM audit_log ORDER BY at DESC LIMIT 5;"
```

- `clients` には、既存の文書・用語集の顧客名が自動で入っています。
- `app_users` には初期管理者が 1 人入っています（`role` が `admin`）。
- `audit_log` には `user_create`（`bootstrap`）が記録されています。

### 6-3. 画面での確認

1. ブラウザで `http://localhost:8501`（NAS の場合は `http://<IPアドレス>:8501`）を開きます。ログイン画面が表示されることを確認します。
2. 初期管理者でログインし、パスワードの変更を求められたら変更します。
3. 「文書管理」で文書件数が移行前と同じであることを確認します。
4. 「過去翻訳を検索」で、既存データが検索できることを確認します。
5. 「用語集」で既存の用語が表示されることを確認します。

> ログインできない場合：`.env` の `INITIAL_ADMIN_LOGIN` / `INITIAL_ADMIN_PASSWORD` を確認し、`docker compose up -d app` で `tm-app` を作り直してください（`.env` は起動時にしか読まれません）。利用者が既に 1 人以上いる場合は、管理者は作られません。その場合は次の手順でパスワードを再発行できます。
>
> ```bash
> docker exec -it tm-app python manage_users.py list
> docker exec -it tm-app python manage_users.py reset-password （管理者のログインID）
> ```

---

## 7. 移行後の初期設定（管理者が画面で実施）

権限の仕組みは、既存のデータにそのまま適用されます。次の順で設定してください。

1. **「顧客・利用者管理」→「顧客」**：自動登録された顧客名を確認します。表記が揺れているものは「名前の変更・統合」で統一します（文書・用語集・担当者がまとめて付け替わります）。
2. **同じ画面で「社外 LLM での翻訳を許可する」を確認**：契約上、社外 AI の利用が制限されている顧客はオフにします。
3. **「利用者」で利用者を追加**：編集者・翻訳者を作成し、担当顧客を設定します。仮パスワードは一度だけ表示されるので、本人に直接伝えます。

注意：

- 担当顧客が設定されていない編集者・翻訳者には、どの文書も見えません。
- 顧客名が空欄の文書は、管理者だけが扱えます。移行前に顧客名を入力しておくと、担当者も扱えるようになります。
- 管理者は常に全顧客を見られます。

---

## 8. ロールバック（元に戻す手順）

移行に問題があった場合の戻し方は、次の 3 段階です。まず A で戻せるかを試し、データに問題がある場合のみ B・C を使います。

### A. コードだけを master に戻す（推奨・データは触らない）

追加されたテーブルは master の旧コードからは参照されないため、データはそのまま使えます。

```bash
cd /Users/solhead/AI/work/rag_translation
git status                       # 変更が無いことを確認
git switch master
docker compose up -d --build
```

作業ツリーの `llm.py` を退避していた場合は、必要に応じて `git stash pop` で戻してください。

### B. DB を手順 2-1 のダンプから復元する

`app` を止めてから、`tm` データベースを作り直して復元します。

```bash
cd /Users/solhead/AI/work/rag_translation
docker compose stop app

docker compose exec -T db psql -U tm -d postgres -c "DROP DATABASE tm WITH (FORCE);"
docker compose exec -T db psql -U tm -d postgres -c "CREATE DATABASE tm;"
docker compose cp "$BACKUP_DIR/tm_full.dump" db:/tmp/tm.dump
docker compose exec db pg_restore -U tm -d tm --no-owner /tmp/tm.dump
```

復元後、手順 1 で控えた件数と一致するか確認してから `docker compose start app` を実行します。

### C. ボリューム全体を手順 2-3 のアーカイブから戻す

最終手段です。`tm-pgdata` の中身を、停止した状態で丸ごと置き換えます。

```bash
cd /Users/solhead/AI/work/rag_translation
docker compose down                       # -v は付けない
docker run --rm \
  -v rag-translation_tm-pgdata:/data \
  -v "$BACKUP_DIR":/backup \
  alpine sh -c "rm -rf /data/* /data/.[!.]* 2>/dev/null; tar xzf /backup/tm-pgdata_volume.tar.gz -C /data"
docker compose up -d
```

---

## 9. 注意事項

- **`docker compose down -v` は実行しない**（データ全削除）。
- **HTTPS の設定を推奨**：ログインのパスワードは、HTTP のままだと LAN 上を平文で流れます。README の「HTTPS（QNAP のリバースプロキシ）」の手順で HTTPS にし、確認後に `.env` の `APP_BIND=127.0.0.1` を設定すると安全です。
- **初期管理者のパスワード**：初回ログイン後に変更し、`.env` の `INITIAL_ADMIN_PASSWORD` は空にしてください。
- **バックアップの保管**：`$BACKUP_DIR` には DB の内容と `.env`（秘密情報）が含まれます。移行が安定するまで保管し、その後は適切に削除してください。
- **ローカル LLM の扱い**：手順 3-1 で退避した `llm.py` の変更（LM Studio / Ollama の有効化）は、`develop-auth1` の意図（現在は使わないためコメントアウト）と異なります。必要であれば、別途ブランチや設定で管理してください。
- 切り替え後の `git` 状態（`develop-auth1`）は、そのまま `git pull` で最新化できます。
