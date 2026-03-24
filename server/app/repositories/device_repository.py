from __future__ import annotations

from datetime import datetime


class DeviceRepository:
    def count_active_devices(self, conn: object, user_id: str) -> int:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS count FROM device_bindings WHERE user_id = %s AND revoked_at IS NULL",
                (user_id,),
            )
            row = cur.fetchone()
        return int(row["count"])

    def get_by_client_key(self, conn: object, user_id: str, client_device_key: str) -> dict | None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    device_id,
                    user_id,
                    client_device_key,
                    platform,
                    device_name,
                    app_version,
                    os_version,
                    last_seen_at,
                    revoked_at,
                    created_at
                FROM device_bindings
                WHERE user_id = %s AND client_device_key = %s
                """,
                (user_id, client_device_key),
            )
            return cur.fetchone()

    def create_device_binding(
        self,
        conn: object,
        *,
        device_id: str,
        user_id: str,
        client_device_key: str,
        platform: str,
        device_name: str,
        app_version: str | None,
        os_version: str | None,
        now: datetime,
    ) -> dict:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO device_bindings (
                    device_id,
                    user_id,
                    client_device_key,
                    platform,
                    device_name,
                    app_version,
                    os_version,
                    last_seen_at,
                    revoked_at,
                    created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULL, %s)
                RETURNING
                    device_id,
                    user_id,
                    client_device_key,
                    platform,
                    device_name,
                    app_version,
                    os_version,
                    last_seen_at,
                    revoked_at,
                    created_at
                """,
                (
                    device_id,
                    user_id,
                    client_device_key,
                    platform,
                    device_name,
                    app_version,
                    os_version,
                    now,
                    now,
                ),
            )
            return cur.fetchone()

    def update_device_binding(
        self,
        conn: object,
        *,
        device_id: str,
        device_name: str,
        app_version: str | None,
        os_version: str | None,
        now: datetime,
    ) -> dict:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE device_bindings
                SET device_name = %s,
                    app_version = %s,
                    os_version = %s,
                    last_seen_at = %s
                WHERE device_id = %s
                RETURNING
                    device_id,
                    user_id,
                    client_device_key,
                    platform,
                    device_name,
                    app_version,
                    os_version,
                    last_seen_at,
                    revoked_at,
                    created_at
                """,
                (device_name, app_version, os_version, now, device_id),
            )
            return cur.fetchone()

    def reactivate_device_binding(
        self,
        conn: object,
        *,
        device_id: str,
        device_name: str,
        app_version: str | None,
        os_version: str | None,
        now: datetime,
    ) -> dict:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE device_bindings
                SET device_name = %s,
                    app_version = %s,
                    os_version = %s,
                    last_seen_at = %s,
                    revoked_at = NULL
                WHERE device_id = %s
                RETURNING
                    device_id,
                    user_id,
                    client_device_key,
                    platform,
                    device_name,
                    app_version,
                    os_version,
                    last_seen_at,
                    revoked_at,
                    created_at
                """,
                (device_name, app_version, os_version, now, device_id),
            )
            return cur.fetchone()

    def list_active_devices(self, conn: object, user_id: str) -> list[dict]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    device_id,
                    user_id,
                    client_device_key,
                    platform,
                    device_name,
                    app_version,
                    os_version,
                    last_seen_at,
                    revoked_at,
                    created_at
                FROM device_bindings
                WHERE user_id = %s AND revoked_at IS NULL
                ORDER BY last_seen_at DESC, created_at DESC
                """,
                (user_id,),
            )
            return list(cur.fetchall())

    def get_owned_device(
        self,
        conn: object,
        *,
        user_id: str,
        device_id: str,
        include_revoked: bool = False,
    ) -> dict | None:
        sql = """
            SELECT
                device_id,
                user_id,
                client_device_key,
                platform,
                device_name,
                app_version,
                os_version,
                last_seen_at,
                revoked_at,
                created_at
            FROM device_bindings
            WHERE user_id = %s AND device_id = %s
        """
        if not include_revoked:
            sql += " AND revoked_at IS NULL"
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, device_id))
            return cur.fetchone()

    def revoke_device(self, conn: object, *, user_id: str, device_id: str, revoked_at: datetime) -> dict | None:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE device_bindings
                SET revoked_at = COALESCE(revoked_at, %s)
                WHERE user_id = %s AND device_id = %s AND revoked_at IS NULL
                RETURNING
                    device_id,
                    user_id,
                    client_device_key,
                    platform,
                    device_name,
                    app_version,
                    os_version,
                    last_seen_at,
                    revoked_at,
                    created_at
                """,
                (revoked_at, user_id, device_id),
            )
            return cur.fetchone()
