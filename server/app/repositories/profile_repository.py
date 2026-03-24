from __future__ import annotations

from datetime import datetime


class ProfileRepository:
    def get_profile_state(self, conn: object, user_id: str, *, for_update: bool = False) -> dict | None:
        sql = """
            SELECT
                profile_id,
                user_id,
                version,
                calibration_offset,
                estimated_accuracy,
                training_count,
                active_model_release_id,
                summary_json,
                updated_at,
                created_at
            FROM profile_states
            WHERE user_id = %s
        """
        if for_update:
            sql += " FOR UPDATE"
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            return cur.fetchone()

    def create_profile_state(self, conn: object, *, profile_id: str, user_id: str, now: datetime) -> dict:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO profile_states (
                    profile_id,
                    user_id,
                    version,
                    calibration_offset,
                    estimated_accuracy,
                    training_count,
                    active_model_release_id,
                    summary_json,
                    updated_at,
                    created_at
                )
                VALUES (
                    %s,
                    %s,
                    1,
                    0,
                    0,
                    0,
                    NULL,
                    '{"retained_feedback_count": 0}'::jsonb,
                    %s,
                    %s
                )
                ON CONFLICT (user_id) DO NOTHING
                RETURNING
                    profile_id,
                    user_id,
                    version,
                    calibration_offset,
                    estimated_accuracy,
                    training_count,
                    active_model_release_id,
                    summary_json,
                    updated_at,
                    created_at
                """,
                (profile_id, user_id, now, now),
            )
            return cur.fetchone()

    def update_profile_state(
        self,
        conn: object,
        *,
        user_id: str,
        current_version: int,
        calibration_offset: float,
        estimated_accuracy: float,
        training_count: int,
        active_model_release_id: str | None,
        summary_json: str,
        updated_at: datetime,
    ) -> dict:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE profile_states
                SET version = %s,
                    calibration_offset = %s,
                    estimated_accuracy = %s,
                    training_count = %s,
                    active_model_release_id = %s,
                    summary_json = %s::jsonb,
                    updated_at = %s
                WHERE user_id = %s AND version = %s
                RETURNING
                    profile_id,
                    user_id,
                    version,
                    calibration_offset,
                    estimated_accuracy,
                    training_count,
                    active_model_release_id,
                    summary_json,
                    updated_at,
                    created_at
                """,
                (
                    current_version + 1,
                    calibration_offset,
                    estimated_accuracy,
                    training_count,
                    active_model_release_id,
                    summary_json,
                    updated_at,
                    user_id,
                    current_version,
                ),
            )
            return cur.fetchone()

    def get_device_calibration(
        self,
        conn: object,
        *,
        user_id: str,
        device_id: str,
        for_update: bool = False,
    ) -> dict | None:
        sql = """
            SELECT
                user_id,
                device_id,
                version,
                device_name,
                sensor_profile_json,
                calibration_offset,
                updated_at,
                created_at
            FROM device_calibrations
            WHERE user_id = %s AND device_id = %s
        """
        if for_update:
            sql += " FOR UPDATE"
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, device_id))
            return cur.fetchone()

    def create_device_calibration(
        self,
        conn: object,
        *,
        user_id: str,
        device_id: str,
        device_name: str,
        now: datetime,
    ) -> dict:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO device_calibrations (
                    user_id,
                    device_id,
                    version,
                    device_name,
                    sensor_profile_json,
                    calibration_offset,
                    updated_at,
                    created_at
                )
                VALUES (%s, %s, 1, %s, '{}'::jsonb, 0, %s, %s)
                ON CONFLICT (user_id, device_id) DO NOTHING
                RETURNING
                    user_id,
                    device_id,
                    version,
                    device_name,
                    sensor_profile_json,
                    calibration_offset,
                    updated_at,
                    created_at
                """,
                (user_id, device_id, device_name, now, now),
            )
            return cur.fetchone()

    def update_device_calibration(
        self,
        conn: object,
        *,
        user_id: str,
        device_id: str,
        current_version: int,
        device_name: str,
        sensor_profile_json: str,
        calibration_offset: float,
        updated_at: datetime,
    ) -> dict:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE device_calibrations
                SET version = %s,
                    device_name = %s,
                    sensor_profile_json = %s::jsonb,
                    calibration_offset = %s,
                    updated_at = %s
                WHERE user_id = %s AND device_id = %s AND version = %s
                RETURNING
                    user_id,
                    device_id,
                    version,
                    device_name,
                    sensor_profile_json,
                    calibration_offset,
                    updated_at,
                    created_at
                """,
                (
                    current_version + 1,
                    device_name,
                    sensor_profile_json,
                    calibration_offset,
                    updated_at,
                    user_id,
                    device_id,
                    current_version,
                ),
            )
            return cur.fetchone()

    def model_release_exists(self, conn: object, model_release_id: str) -> bool:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM model_releases WHERE model_release_id = %s",
                (model_release_id,),
            )
            return cur.fetchone() is not None
