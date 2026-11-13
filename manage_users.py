"""利用者の管理（コマンド）。管理者がログインできなくなったときの復旧用。

QNAP 上で、アプリのコンテナの中で実行する:

    docker exec -it tm-app python manage_users.py list
    docker exec -it tm-app python manage_users.py reset-password admin@example.co.jp
    docker exec -it tm-app python manage_users.py create-admin admin2@example.co.jp "管理者 二郎"

reset-password と create-admin は仮パスワードを表示する。初回ログイン時に変更を求める。
"""
from __future__ import annotations

import argparse
import sys

from auth import audit, create_user, ensure_auth_tables, find_user_by_login, generate_password, list_users, \
    reset_password
from common import connect


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG Translation App の利用者管理（復旧用）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="利用者の一覧")
    p = sub.add_parser("reset-password", help="仮パスワードを発行し、ロックを解除する")
    p.add_argument("login_id")
    p = sub.add_parser("create-admin", help="管理者を追加する")
    p.add_argument("login_id")
    p.add_argument("name")
    args = parser.parse_args()

    with connect() as conn:
        ensure_auth_tables(conn)
        if args.cmd == "list":
            for u in list_users(conn):
                state = "有効" if u["active"] else "無効"
                print(f"{u['login_id']}\t{u['display_name']}\t{u['role']}\t{state}")
            return 0
        if args.cmd == "reset-password":
            row = conn.execute("SELECT id FROM app_users WHERE lower(login_id) = lower(%s)",
                               (args.login_id,)).fetchone()
            if row is None:
                print(f"利用者「{args.login_id}」が見つかりません。", file=sys.stderr)
                return 1
            pw = generate_password()
            reset_password(conn, row[0], pw)
            conn.execute("UPDATE app_users SET active = true WHERE id = %s", (row[0],))
            audit(conn, None, "password_reset", target=args.login_id, detail={"by": "manage_users.py"})
            conn.commit()
            print(f"仮パスワード: {pw}")
            return 0
        if args.cmd == "create-admin":
            if find_user_by_login(conn, args.login_id):
                print(f"利用者「{args.login_id}」は登録済みです。", file=sys.stderr)
                return 1
            pw = generate_password()
            create_user(conn, args.login_id, args.name, "admin", all_clients=True, clients=[], password=pw)
            audit(conn, None, "user_create", target=args.login_id, detail={"role": "admin", "by": "manage_users.py"})
            conn.commit()
            print(f"仮パスワード: {pw}")
            return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
