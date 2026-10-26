# RAGによる前年踏襲翻訳（Streamlit）

翻訳作業において、前年（過去）と同じ文章や類似した表現をRAG（検索拡張生成：Retrieval-Augmented Generation）を活用して自動的に置き換えるアプリです。

従来の翻訳メモリ（TM: Translation Memory）の仕組みを、LLMとベクトル検索を使って高度化させたものです。

## 手順

1. idml2xlsxを使って英語と和文のIDMLファイルからテキスト情報を取り出して対訳表エクセルを作成
2. Docker Desktop（Windows）または Container Station（QNAP）で、DBとアプリをまとめて起動
    ```bash
    # 初回のみ：.env.example を .env にコピーし、POSTGRES_PASSWORD を英数字のパスワードに変更
    #           （同梱の .env には自動生成したパスワードが設定済み）
    docker compose up -d --build
    ```
    - DB（`tm-pg`）は初回起動時に `schema.sql` を自動実行してテーブルを作成します。
    - アプリ（`tm-app`）は DB の準備完了を待ってから起動します。
    - 状態確認：`docker compose ps` ／ ログ：`docker compose logs -f`
3. ブラウザで `http://localhost:8501`（NAS の場合は `http://<NASのIP>:8501`）を開く
4. 「データ取り込み」タブで対訳表エクセルをアップロードし、「取り込む」ボタンで一括データベース化
5. アプリのテキストボックスに検索したい英文を貼り付け「検索」ボタンを押すと一致度順に過去英文とその和訳、入力英文と過去英文との差分が表示される

## 仕組み
- **文（文脈）の切り出し（Chunking）:** 前年と同じ文を見つけるため、段落ごとではなく、文章を「1文ごと（センテンス単位）」に分割。
- **ベクトル化（Embedding）:** 前年の「原文」をEmbeddingモデル（`multilingual-e5-small`=384次元）を使ってベクトルデータに変換。
- **ベクトルDBへの格納:** ベクトル化した「原文」と、それに対応する「前年の訳文」をセット（メタデータ）としてベクトルデータベース（PostgreSQL）に保存。
- **検索ロジック** 意味的な「ベクトル類似」に加え、文字列同士の似ている度合い「トライグラム類似」、「完全一致」の３つの方法でハイブリッド検索。
- **差分表示** 検索順に表示し、それぞれ検索文との差分を表示します。

## 特長
- 従来の「翻訳メモリ（TM）」と違い、「完全に同じではないけれど、少しだけ単語が変わった文章」に対しても、前年のトーンを維持したまま柔軟に翻訳できる

## 今後の開発方針
- 翻訳LLMに原文に加え、前年の原文と差分、前年の訳文を含めて投げることで、前年の訳例のスタイルや用語、トーンを厳密に踏襲させる。
- 用語集（グロッサリー）との併用で固有名詞、専門用語、社内用語をより厳格に反映することができる。


## Docker 運用メモ
| 操作 | コマンド |
|---|---|
| 停止 | `docker compose down` |
| コード変更後の再起動 | `docker compose up -d --build` |
| DB を初期化してやり直す（**データ全削除**） | `docker compose down -v` |
| psql で DB に入る | `docker compose exec db psql -U tm -d tm` |
| CSV を CLI で一括投入 | `docker compose cp data.csv app:/tmp/data.csv` → `docker compose exec app python ingest.py /tmp/data.csv --defer-index` |

- 以前 `docker run` で作った `tm-pg` コンテナが残っていると名前が重なるため、先に `docker rm -f tm-pg` で削除してください。
- 埋め込みモデル（multilingual-e5-small）はビルド時にイメージへ同梱されるため、起動後はインターネット接続不要です。初回ビルドは数分〜十数分かかります。
- DB のポートは同じPC内（127.0.0.1）からのみ接続できます。アプリ画面（8501）は社内LANに公開されるため、ルーターでポート開放しないでください。
