"""ログイン（ユーザー認証）・権限・顧客マスター・操作ログ。

画面（Streamlit）に依存しない部分だけをこのファイルに置く。将来 FastAPI などに移る場合も
そのまま使えるようにするため。画面のログイン処理は app.py の require_login() にある。

テーブル（アプリ起動時に自動で作成。既存のテーブルは変更しない）
    clients       顧客の正式な一覧（顧客名の表記を統一する）と、社外 LLM の利用可否
    app_users     利用者（ログイン ID・パスワードのハッシュ・役割・全顧客かどうか など）
    user_clients  利用者の担当顧客
    audit_log     操作ログ（誰が・いつ・何をしたか。翻訳した文章そのものは記録しない）

役割
    admin       管理者：すべての操作。全顧客。利用者・顧客の管理、操作ログの閲覧
    editor      編集者：検索・翻訳に加え、取り込み・文書管理・顧客の用語集の編集（担当顧客のみ）
    translator  翻訳者：検索・翻訳・用語集の閲覧（担当顧客のみ）

認証方式（環境変数 AUTH_MODE）
    local  アプリ内のログイン ID とパスワード（既定）
    oidc   社内アカウントでのシングルサインオン（Streamlit の st.login。Microsoft Entra ID など）。
           ログインした人のメールアドレスが app_users.login_id と一致し、有効な場合だけ使える。
           役割・担当顧客はこれまでどおりアプリ側で管理する。
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import string
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

ROLES = {"admin": "管理者", "editor": "編集者", "translator": "翻訳者"}
AUTH_MODE = os.getenv("AUTH_MODE", "local").strip().lower()
MAX_FAILED_LOGINS = int(os.getenv("AUTH_MAX_FAILED", "5"))          # この回数失敗したらロック
LOCK_MINUTES = int(os.getenv("AUTH_LOCK_MINUTES", "15"))            # ロックする時間（分）
SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "60"))  # 無操作で自動ログアウト
MIN_PASSWORD_LENGTH = 10

AUTH_DDL = """
CREATE TABLE IF NOT EXISTS clients (
    name         text        PRIMARY KEY,                 -- documents.client / glossary_terms.client と同じ表記
    llm_allowed  boolean     NOT NULL DEFAULT true,       -- false = 社外 LLM での翻訳・用語抽出を禁止
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS app_users (
    id                   bigserial   PRIMARY KEY,
    login_id             text        NOT NULL,            -- 社内メールアドレスを推奨（シングルサインオンで照合）
    display_name         text        NOT NULL,
    password_hash        text,                            -- Argon2。シングルサインオンだけの利用者は空
    role                 text        NOT NULL CHECK (role IN ('admin', 'editor', 'translator')),
    all_clients          boolean     NOT NULL DEFAULT false,
    active               boolean     NOT NULL DEFAULT true,
    must_change_password boolean     NOT NULL DEFAULT true,
    failed_logins        int         NOT NULL DEFAULT 0,
    locked_until         timestamptz,
    last_login_at        timestamptz,
    created_at           timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS app_users_login_id_key ON app_users (lower(login_id));
CREATE TABLE IF NOT EXISTS user_clients (
    user_id  bigint NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,
    client   text   NOT NULL REFERENCES clients(name) ON UPDATE CASCADE ON DELETE CASCADE,
    PRIMARY KEY (user_id, client)
);
CREATE TABLE IF NOT EXISTS audit_log (
    id        bigserial   PRIMARY KEY,
    at        timestamptz NOT NULL DEFAULT now(),
    user_id   bigint      REFERENCES app_users(id) ON DELETE SET NULL,
    login_id  text,                                       -- 利用者を消しても誰の操作か分かるように残す
    action    text        NOT NULL,
    target    text,
    detail    jsonb
);
CREATE INDEX IF NOT EXISTS idx_audit_log_at ON audit_log (at DESC);
"""

ACTION_LABELS = {
    "login": "ログイン", "login_failed": "ログイン失敗", "logout": "ログアウト",
    "session_timeout": "自動ログアウト", "password_change": "パスワード変更",
    "ingest": "対訳表の取り込み", "update_document": "文書情報の修正", "delete_document": "文書の削除",
    "glossary_save": "用語集の保存", "glossary_import": "用語集の取り込み", "glossary_delete": "用語集の削除",
    "glossary_extract_import": "抽出した用語の取り込み", "translate": "翻訳の実行", "usage_reset": "トークン数の初期化",
    "user_create": "利用者の追加", "user_update": "利用者の変更", "password_reset": "パスワードの再設定",
    "client_create": "顧客の追加", "client_update": "顧客の設定変更", "client_rename": "顧客名の変更・統合",
    "client_delete": "顧客の削除",
}

_log = logging.getLogger(__name__)
_hasher = PasswordHasher()


# ---------------------------------------------------------------------
# 利用者
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class User:
    id: int
    login_id: str
    display_name: str
    role: str
    all_clients: bool
    clients: tuple[str, ...] = field(default_factory=tuple)
    must_change_password: bool = False

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def can_edit(self) -> bool:
        """取り込み・文書管理・顧客の用語集の編集ができるか（管理者・編集者）。"""
        return self.role in ("admin", "editor")

    @property
    def role_label(self) -> str:
        return ROLES.get(self.role, self.role)

    def visible_clients(self, conn) -> list[str] | None:
        """見られる顧客名。None は制限なし（管理者：顧客名が空欄の文書も含む）。

        「全顧客」の編集者・翻訳者は、登録済みの全顧客（顧客名が空欄の文書は含まない）。
        """
        if self.is_admin:
            return None
        if self.all_clients:
            return list_client_names(conn)
        return list(self.clients)

    def can_see_client(self, conn, client: str | None) -> bool:
        scope = self.visible_clients(conn)
        return scope is None or (client is not None and client in scope)


def _user_from_row(conn, row) -> User:
    uid, login_id, name, role, all_clients, must_change = row
    clients = tuple(r[0] for r in conn.execute(
        "SELECT client FROM user_clients WHERE user_id = %s ORDER BY client", (uid,)))
    return User(uid, login_id, name, role, all_clients, clients, must_change)


_USER_COLS = "id, login_id, display_name, role, all_clients, must_change_password"


def load_user(conn, user_id: int) -> User | None:
    """有効な利用者を読み込む（無効化・削除されていれば None）。毎回の画面表示で呼び、変更をすぐ反映する。"""
    row = conn.execute(f"SELECT {_USER_COLS} FROM app_users WHERE id = %s AND active", (user_id,)).fetchone()
    return _user_from_row(conn, row) if row else None


def find_user_by_login(conn, login_id: str) -> User | None:
    row = conn.execute(f"SELECT {_USER_COLS} FROM app_users WHERE lower(login_id) = lower(%s) AND active",
                       ((login_id or "").strip(),)).fetchone()
    return _user_from_row(conn, row) if row else None


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def check_password_rule(password: str) -> str | None:
    """パスワードの条件を満たさない場合は理由を返す。"""
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"パスワードは {MIN_PASSWORD_LENGTH} 文字以上にしてください。"
    kinds = sum(any(f(c) for c in password) for f in (str.isupper, str.islower, str.isdigit))
    if kinds < 2:
        return "パスワードには英大文字・英小文字・数字のうち 2 種類以上を含めてください。"
    return None


def generate_password(length: int = 14) -> str:
    """仮パスワード（紛らわしい文字を除いた英数字）。"""
    alphabet = "".join(c for c in string.ascii_letters + string.digits if c not in "Il1O0o")
    while True:
        pw = "".join(secrets.choice(alphabet) for _ in range(length))
        if check_password_rule(pw) is None:
            return pw


def authenticate(conn, login_id: str, password: str) -> tuple[User | None, str]:
    """ログイン ID とパスワードを照合する。(利用者, メッセージ) を返す。失敗の理由は区別して見せない。"""
    generic = "ログイン ID またはパスワードが違います。"
    row = conn.execute(
        "SELECT id, password_hash, active, failed_logins, locked_until FROM app_users"
        " WHERE lower(login_id) = lower(%s)", ((login_id or "").strip(),)).fetchone()
    if row is None:
        _hasher.hash(password or "x")  # 存在しない ID でも処理時間をそろえる
        audit(conn, None, "login_failed", login_id=(login_id or "")[:100], detail={"reason": "unknown_id"})
        return None, generic
    uid, pw_hash, active, failed, locked_until = row
    now = datetime.now(timezone.utc)
    if locked_until and locked_until > now:
        audit(conn, None, "login_failed", login_id=login_id, detail={"reason": "locked"}, user_id=uid)
        minutes = max(1, int((locked_until - now).total_seconds() // 60) + 1)
        return None, f"ログインに続けて失敗したため、ロックしています。約 {minutes} 分後に再度お試しください。"
    ok = False
    if active and pw_hash:
        try:
            ok = _hasher.verify(pw_hash, password or "")
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            ok = False
    if not ok:
        failed += 1
        lock = now + timedelta(minutes=LOCK_MINUTES) if failed >= MAX_FAILED_LOGINS else None
        conn.execute("UPDATE app_users SET failed_logins = %s, locked_until = %s WHERE id = %s",
                     (0 if lock else failed, lock, uid))
        audit(conn, None, "login_failed", login_id=login_id, user_id=uid,
              detail={"reason": "inactive" if not active else "password", "locked": bool(lock)})
        conn.commit()
        return None, generic
    if _hasher.check_needs_rehash(pw_hash):
        conn.execute("UPDATE app_users SET password_hash = %s WHERE id = %s", (hash_password(password), uid))
    conn.execute("UPDATE app_users SET failed_logins = 0, locked_until = NULL, last_login_at = now()"
                 " WHERE id = %s", (uid,))
    user = load_user(conn, uid)
    audit(conn, user, "login")
    conn.commit()
    return user, ""


def record_sso_login(conn, user: User) -> None:
    conn.execute("UPDATE app_users SET last_login_at = now() WHERE id = %s", (user.id,))
    audit(conn, user, "login", detail={"method": "oidc"})
    conn.commit()


def change_password(conn, user_id: int, new_password: str) -> None:
    conn.execute("UPDATE app_users SET password_hash = %s, must_change_password = false WHERE id = %s",
                 (hash_password(new_password), user_id))
    conn.commit()


def verify_current_password(conn, user_id: int, password: str) -> bool:
    row = conn.execute("SELECT password_hash FROM app_users WHERE id = %s", (user_id,)).fetchone()
    if not row or not row[0]:
        return False
    try:
        return _hasher.verify(row[0], password or "")
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


# ---------------------------------------------------------------------
# 利用者の管理（管理者）
# ---------------------------------------------------------------------
def list_users(conn) -> list[dict]:
    cur = conn.execute(
        """
        SELECT u.id, u.login_id, u.display_name, u.role, u.all_clients, u.active,
               u.must_change_password, u.failed_logins, u.locked_until, u.last_login_at, u.created_at,
               coalesce(array_agg(uc.client ORDER BY uc.client) FILTER (WHERE uc.client IS NOT NULL), '{}')
                 AS clients
          FROM app_users u LEFT JOIN user_clients uc ON uc.user_id = u.id
         GROUP BY u.id ORDER BY u.active DESC, u.role, u.login_id
        """)
    cols = [c.name for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def create_user(conn, login_id: str, display_name: str, role: str, *, all_clients: bool,
                clients: list[str], password: str | None) -> int:
    login_id, display_name = (login_id or "").strip(), (display_name or "").strip()
    if not login_id or not display_name:
        raise ValueError("ログイン ID と氏名は必須です。")
    if role not in ROLES:
        raise ValueError("役割が正しくありません。")
    with conn.transaction():
        uid = conn.execute(
            "INSERT INTO app_users (login_id, display_name, password_hash, role, all_clients, must_change_password)"
            " VALUES (%s, %s, %s, %s, %s, true) RETURNING id",
            (login_id, display_name, hash_password(password) if password else None, role,
             all_clients or role == "admin")).fetchone()[0]
        _set_user_clients(conn, uid, clients)
    return uid


def update_user(conn, user_id: int, *, display_name: str, role: str, all_clients: bool,
                clients: list[str], active: bool) -> None:
    if role not in ROLES:
        raise ValueError("役割が正しくありません。")
    with conn.transaction():
        if role != "admin" or not active:
            _ensure_other_admin(conn, user_id)
        conn.execute("UPDATE app_users SET display_name = %s, role = %s, all_clients = %s, active = %s"
                     " WHERE id = %s", (display_name.strip(), role, all_clients or role == "admin", active, user_id))
        _set_user_clients(conn, user_id, clients)


def _ensure_other_admin(conn, user_id: int) -> None:
    """有効な管理者が 1 人もいなくなる変更を防ぐ。"""
    n = conn.execute("SELECT count(*) FROM app_users WHERE role = 'admin' AND active AND id <> %s",
                     (user_id,)).fetchone()[0]
    if n == 0:
        raise ValueError("有効な管理者が 1 人もいなくなるため、変更できません。先に別の管理者を登録してください。")


def _set_user_clients(conn, user_id: int, clients: list[str]) -> None:
    conn.execute("DELETE FROM user_clients WHERE user_id = %s", (user_id,))
    for c in sorted(set(clients)):
        conn.execute("INSERT INTO user_clients (user_id, client) VALUES (%s, %s)", (user_id, c))


def reset_password(conn, user_id: int, password: str) -> None:
    """管理者による再設定。次回ログイン時に本人に変更させ、ロックも解除する。"""
    conn.execute("UPDATE app_users SET password_hash = %s, must_change_password = true, failed_logins = 0,"
                 " locked_until = NULL WHERE id = %s", (hash_password(password), user_id))
    conn.commit()


def unlock_user(conn, user_id: int) -> None:
    conn.execute("UPDATE app_users SET failed_logins = 0, locked_until = NULL WHERE id = %s", (user_id,))
    conn.commit()


# ---------------------------------------------------------------------
# 顧客マスター
# ---------------------------------------------------------------------
def list_client_names(conn) -> list[str]:
    return [r[0] for r in conn.execute("SELECT name FROM clients ORDER BY name")]


def llm_forbidden_clients(conn) -> list[str]:
    """社外 LLM を使わない設定の顧客。"""
    return [r[0] for r in conn.execute("SELECT name FROM clients WHERE NOT llm_allowed ORDER BY name")]


def list_clients_detail(conn) -> list[dict]:
    cur = conn.execute(
        """
        SELECT c.name, c.llm_allowed,
               (SELECT count(*) FROM documents d WHERE d.client = c.name) AS documents,
               (SELECT count(*) FROM glossary_terms g WHERE g.client = c.name) AS terms,
               (SELECT count(*) FROM user_clients uc WHERE uc.client = c.name) AS users
          FROM clients c ORDER BY c.name
        """)
    cols = [c.name for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def create_client(conn, name: str, llm_allowed: bool = True) -> str:
    from documents import clean_field
    name = clean_field(name)
    if not name:
        raise ValueError("顧客名を入力してください。")
    if conn.execute("SELECT 1 FROM clients WHERE name = %s", (name,)).fetchone():
        raise ValueError(f"顧客「{name}」は登録済みです。")
    conn.execute("INSERT INTO clients (name, llm_allowed) VALUES (%s, %s)", (name, llm_allowed))
    conn.commit()
    return name


def set_client_llm_allowed(conn, name: str, allowed: bool) -> None:
    conn.execute("UPDATE clients SET llm_allowed = %s WHERE name = %s", (allowed, name))
    conn.commit()


def rename_client(conn, old: str, new: str) -> str:
    """顧客名を変更する。new が登録済みなら old を new に統合する（文書・用語集・担当者を付け替える）。

    統合時、同じ用語（顧客・媒体・英語・日本語が同じ）は new 側を残す。社外 LLM の設定は、どちらかが
    「使わない」なら「使わない」にする（安全側）。
    """
    from documents import clean_field
    new = clean_field(new)
    if not new:
        raise ValueError("新しい顧客名を入力してください。")
    if new == old:
        raise ValueError("顧客名が変わっていません。")
    with conn.transaction():
        src = conn.execute("SELECT llm_allowed FROM clients WHERE name = %s", (old,)).fetchone()
        if src is None:
            raise ValueError(f"顧客「{old}」が見つかりません。")
        dst = conn.execute("SELECT llm_allowed FROM clients WHERE name = %s", (new,)).fetchone()
        if dst is None:
            conn.execute("UPDATE clients SET name = %s WHERE name = %s", (new, old))  # user_clients も連動
        else:
            conn.execute("UPDATE clients SET llm_allowed = %s WHERE name = %s", (src[0] and dst[0], new))
            conn.execute("INSERT INTO user_clients (user_id, client) SELECT user_id, %s FROM user_clients"
                         " WHERE client = %s ON CONFLICT DO NOTHING", (new, old))
            conn.execute("DELETE FROM clients WHERE name = %s", (old,))
        conn.execute("UPDATE documents SET client = %s WHERE client = %s", (new, old))
        conn.execute(
            """
            INSERT INTO glossary_terms (client, medium, en, ja, note, required)
            SELECT %s, medium, en, ja, note, required FROM glossary_terms WHERE client = %s
            ON CONFLICT (client, medium, en, ja) DO NOTHING
            """, (new, old))
        conn.execute("DELETE FROM glossary_terms WHERE client = %s", (old,))
    return new


def delete_client(conn, name: str) -> None:
    """使われていない顧客だけ削除できる（文書・用語集が残っている場合は先に統合・削除する）。"""
    with conn.transaction():
        used = conn.execute(
            "SELECT (SELECT count(*) FROM documents WHERE client = %s) + "
            "(SELECT count(*) FROM glossary_terms WHERE client = %s)", (name, name)).fetchone()[0]
        if used:
            raise ValueError("文書または用語集が残っているため削除できません。別の顧客に統合してください。")
        conn.execute("DELETE FROM clients WHERE name = %s", (name,))


# ---------------------------------------------------------------------
# 操作ログ
# ---------------------------------------------------------------------
def audit(conn, user: User | None, action: str, *, target: str | None = None,
          detail: dict | None = None, user_id: int | None = None, login_id: str | None = None) -> None:
    """操作ログを 1 件書く。記録に失敗しても操作は止めない（警告だけ出す）。"""
    try:
        conn.execute(
            "INSERT INTO audit_log (user_id, login_id, action, target, detail) VALUES (%s, %s, %s, %s, %s)",
            (user.id if user else user_id, user.login_id if user else login_id, action, target,
             json.dumps(detail, ensure_ascii=False, default=str) if detail else None))
    except Exception as exc:  # noqa: BLE001
        _log.warning("操作ログの記録に失敗しました（%s）: %s", action, exc)


def list_audit_log(conn, *, limit: int = 500, login_id: str | None = None,
                   actions: list[str] | None = None) -> list[dict]:
    where, params = [], []
    if login_id:
        where.append("login_id = %s")
        params.append(login_id)
    if actions:
        where.append("action = ANY(%s)")
        params.append(actions)
    sql = ("SELECT at, login_id, action, target, detail FROM audit_log"
           + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY at DESC, id DESC LIMIT %s")
    cur = conn.execute(sql, [*params, limit])
    cols = [c.name for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


# ---------------------------------------------------------------------
# テーブルの作成と初期設定
# ---------------------------------------------------------------------
def ensure_auth_tables(conn) -> None:
    """テーブルが無ければ作り、既存の顧客名を顧客マスターに登録し、初めの管理者を作る。"""
    conn.execute(AUTH_DDL)
    # 既存の文書・用語集に使われている顧客名を顧客マスターに登録（顧客名の統一は画面で行う）
    conn.execute(
        """
        INSERT INTO clients (name)
        SELECT client FROM documents WHERE client IS NOT NULL AND client <> ''
        UNION
        SELECT client FROM glossary_terms WHERE client <> ''
        ON CONFLICT DO NOTHING
        """)
    bootstrap_admin(conn)
    conn.commit()


def bootstrap_admin(conn) -> bool:
    """利用者が 1 人もいない場合だけ、.env の INITIAL_ADMIN_LOGIN / INITIAL_ADMIN_PASSWORD から管理者を作る。"""
    if conn.execute("SELECT count(*) FROM app_users").fetchone()[0]:
        return False
    login = os.getenv("INITIAL_ADMIN_LOGIN", "").strip()
    password = os.getenv("INITIAL_ADMIN_PASSWORD", "")
    if not login or not password:
        return False
    conn.execute(
        "INSERT INTO app_users (login_id, display_name, password_hash, role, all_clients, must_change_password)"
        " VALUES (%s, %s, %s, 'admin', true, true)",
        (login, os.getenv("INITIAL_ADMIN_NAME", "管理者"), hash_password(password)))
    audit(conn, None, "user_create", target=login, detail={"role": "admin", "bootstrap": True})
    return True


def has_users(conn) -> bool:
    return bool(conn.execute("SELECT count(*) FROM app_users").fetchone()[0])
