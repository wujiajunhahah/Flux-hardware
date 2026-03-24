from __future__ import annotations

from datetime import datetime


class AuthRepository:
    def get_identity(self, conn: object, provider: str, provider_subject: str) -> dict | None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ai.identity_id, ai.user_id, u.status, u.primary_email
                FROM auth_identities ai
                JOIN users u ON u.user_id = ai.user_id
                WHERE ai.provider = %s AND ai.provider_subject = %s
                """,
                (provider, provider_subject),
            )
            return cur.fetchone()

    def lock_user(self, conn: object, user_id: str) -> None:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM users WHERE user_id = %s FOR UPDATE", (user_id,))

    def create_user(
        self,
        conn: object,
        *,
        user_id: str,
        primary_email: str | None,
        timezone: str | None,
        now: datetime,
    ) -> dict:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (user_id, status, primary_email, timezone, created_at, updated_at)
                VALUES (%s, 'active', %s, %s, %s, %s)
                RETURNING user_id, status, primary_email, timezone, created_at, updated_at
                """,
                (user_id, primary_email, timezone, now, now),
            )
            return cur.fetchone()

    def create_identity(
        self,
        conn: object,
        *,
        identity_id: str,
        user_id: str,
        provider: str,
        provider_subject: str,
        now: datetime,
    ) -> dict | None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO auth_identities (
                    identity_id,
                    user_id,
                    provider,
                    provider_subject,
                    created_at
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (provider, provider_subject) DO NOTHING
                RETURNING identity_id, user_id, provider, provider_subject, created_at
                """,
                (identity_id, user_id, provider, provider_subject, now),
            )
            return cur.fetchone()

    def create_refresh_token(
        self,
        conn: object,
        *,
        token_id: str,
        user_id: str,
        device_id: str,
        token_hash: str,
        expires_at: datetime,
        now: datetime,
    ) -> dict:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO refresh_tokens (
                    token_id,
                    user_id,
                    device_id,
                    token_hash,
                    expires_at,
                    revoked_at,
                    created_at
                )
                VALUES (%s, %s, %s, %s, %s, NULL, %s)
                RETURNING token_id, user_id, device_id, expires_at, revoked_at, created_at
                """,
                (token_id, user_id, device_id, token_hash, expires_at, now),
            )
            return cur.fetchone()

    def get_refresh_token(self, conn: object, token_hash: str) -> dict | None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    rt.token_id,
                    rt.user_id,
                    rt.device_id,
                    rt.expires_at,
                    rt.revoked_at,
                    u.status AS user_status,
                    db.revoked_at AS device_revoked_at
                FROM refresh_tokens rt
                JOIN users u ON u.user_id = rt.user_id
                JOIN device_bindings db ON db.device_id = rt.device_id
                WHERE rt.token_hash = %s
                """,
                (token_hash,),
            )
            return cur.fetchone()

    def revoke_refresh_token(self, conn: object, token_hash: str, revoked_at: datetime) -> dict | None:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE refresh_tokens
                SET revoked_at = COALESCE(revoked_at, %s)
                WHERE token_hash = %s
                RETURNING token_id, user_id, device_id, token_hash, expires_at, revoked_at, created_at
                """,
                (revoked_at, token_hash),
            )
            return cur.fetchone()

    def revoke_refresh_tokens_for_device(
        self,
        conn: object,
        *,
        user_id: str,
        device_id: str,
        revoked_at: datetime,
    ) -> int:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE refresh_tokens
                SET revoked_at = COALESCE(revoked_at, %s)
                WHERE user_id = %s AND device_id = %s
                """,
                (revoked_at, user_id, device_id),
            )
            return cur.rowcount

    def get_active_principal(self, conn: object, user_id: str, device_id: str) -> dict | None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT u.user_id, db.device_id
                FROM users u
                JOIN device_bindings db ON db.user_id = u.user_id
                WHERE u.user_id = %s
                  AND u.status = 'active'
                  AND db.device_id = %s
                  AND db.revoked_at IS NULL
                """,
                (user_id, device_id),
            )
            return cur.fetchone()
