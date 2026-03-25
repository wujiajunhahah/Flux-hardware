from __future__ import annotations


MODEL_COLUMNS = """
    model_release_id,
    model_kind,
    platform,
    channel,
    version,
    artifact_object_key,
    artifact_sha256,
    config_json,
    status,
    published_at,
    created_at
"""


class ModelRepository:
    def get_model_release(self, conn: object, model_release_id: str) -> dict | None:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {MODEL_COLUMNS} FROM model_releases WHERE model_release_id = %s",
                (model_release_id,),
            )
            return cur.fetchone()

    def get_latest_published_model_release(
        self,
        conn: object,
        *,
        platform: str,
        channel: str,
        model_kind: str = "global_base_model",
    ) -> dict | None:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {MODEL_COLUMNS}
                FROM model_releases
                WHERE platform = %s
                  AND channel = %s
                  AND model_kind = %s
                  AND status = 'published'
                ORDER BY published_at DESC NULLS LAST, created_at DESC, model_release_id DESC
                LIMIT 1
                """,
                (platform, channel, model_kind),
            )
            return cur.fetchone()
