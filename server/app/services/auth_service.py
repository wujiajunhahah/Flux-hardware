from __future__ import annotations

from datetime import timedelta

from ..core.db import connection
from ..core.errors import AppError
from ..core.identity import resolve_provider_identity
from ..core.ids import new_id
from ..core.security import build_access_token, generate_refresh_token, hash_token, utcnow
from ..core.settings import settings
from ..domain.auth import DevicePayload, RefreshRequest, SignInRequest, SignOutRequest
from ..repositories.auth_repository import AuthRepository
from ..repositories.device_repository import DeviceRepository
from ..repositories.profile_repository import ProfileRepository


class AuthService:
    def __init__(
        self,
        auth_repository: AuthRepository | None = None,
        device_repository: DeviceRepository | None = None,
        profile_repository: ProfileRepository | None = None,
    ) -> None:
        self.auth_repository = auth_repository or AuthRepository()
        self.device_repository = device_repository or DeviceRepository()
        self.profile_repository = profile_repository or ProfileRepository()

    def sign_up(self, payload: SignInRequest) -> dict:
        return self._authenticate(payload, create_user=True)

    def sign_in(self, payload: SignInRequest) -> dict:
        return self._authenticate(payload, create_user=False)

    def _authenticate(self, payload: SignInRequest, *, create_user: bool) -> dict:
        resolved_identity = resolve_provider_identity(payload.provider, payload.provider_token)
        now = utcnow()
        refresh_token = generate_refresh_token()
        refresh_expires_at = now + timedelta(seconds=settings.refresh_token_ttl_sec)

        with connection() as conn:
            try:
                identity_row = self.auth_repository.get_identity(
                    conn,
                    resolved_identity.provider,
                    resolved_identity.provider_subject,
                )

                if create_user:
                    if identity_row is not None:
                        raise AppError(409, "identity_already_exists", "identity already exists")
                    user_id = new_id("usr")
                    self.auth_repository.create_user(
                        conn,
                        user_id=user_id,
                        primary_email=resolved_identity.primary_email,
                        timezone=None,
                        now=now,
                    )
                    created_identity = self.auth_repository.create_identity(
                        conn,
                        identity_id=new_id("idn"),
                        user_id=user_id,
                        provider=resolved_identity.provider,
                        provider_subject=resolved_identity.provider_subject,
                        now=now,
                    )
                    if created_identity is None:
                        raise AppError(409, "identity_already_exists", "identity already exists")
                    self.auth_repository.lock_user(conn, user_id)
                    self._ensure_profile(conn, user_id, now)
                else:
                    if identity_row is None:
                        raise AppError(404, "user_not_found", "user not found")
                    if identity_row["status"] != "active":
                        raise AppError(403, "forbidden", "user is not active")
                    user_id = identity_row["user_id"]
                    self.auth_repository.lock_user(conn, user_id)
                    self._ensure_profile(conn, user_id, now)

                device = self._reconcile_device(conn, user_id=user_id, device=payload.device, now=now)
                self._ensure_device_calibration(
                    conn,
                    user_id=user_id,
                    device_id=device["device_id"],
                    device_name=device["device_name"],
                    now=now,
                )
                self.auth_repository.create_refresh_token(
                    conn,
                    token_id=new_id("rtk"),
                    user_id=user_id,
                    device_id=device["device_id"],
                    token_hash=hash_token(refresh_token),
                    expires_at=refresh_expires_at,
                    now=now,
                )
                conn.commit()
            except AppError:
                conn.rollback()
                raise
            except Exception:
                conn.rollback()
                raise

        access_token = build_access_token(user_id=user_id, device_id=device["device_id"])
        return {
            "user_id": user_id,
            "device_id": device["device_id"],
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in_sec": settings.access_token_ttl_sec,
        }

    def refresh_access_token(self, payload: RefreshRequest) -> dict:
        now = utcnow()
        token_hash = hash_token(payload.refresh_token)
        with connection() as conn:
            refresh_row = self.auth_repository.get_refresh_token(conn, token_hash)

        if refresh_row is None:
            raise AppError(401, "unauthorized", "refresh token is invalid")
        if refresh_row["revoked_at"] is not None:
            raise AppError(401, "unauthorized", "refresh token is invalid")
        if refresh_row["device_revoked_at"] is not None or refresh_row["user_status"] != "active":
            raise AppError(401, "unauthorized", "refresh token is invalid")
        if refresh_row["expires_at"] <= now:
            raise AppError(401, "unauthorized", "refresh token is expired")

        return {
            "access_token": build_access_token(
                user_id=refresh_row["user_id"],
                device_id=refresh_row["device_id"],
            ),
            "expires_in_sec": settings.access_token_ttl_sec,
        }

    def sign_out(self, payload: SignOutRequest) -> dict:
        now = utcnow()
        token_hash = hash_token(payload.refresh_token)

        with connection() as conn:
            try:
                revoked = self.auth_repository.revoke_refresh_token(conn, token_hash, now)
                if revoked is None:
                    raise AppError(401, "unauthorized", "refresh token is invalid")
                conn.commit()
            except AppError:
                conn.rollback()
                raise
            except Exception:
                conn.rollback()
                raise

        return {"revoked": True}

    def _ensure_profile(self, conn: object, user_id: str, now: object) -> dict:
        profile = self.profile_repository.get_profile_state(conn, user_id)
        if profile is not None:
            return profile
        created = self.profile_repository.create_profile_state(
            conn,
            profile_id=new_id("pro"),
            user_id=user_id,
            now=now,
        )
        if created is not None:
            return created
        return self.profile_repository.get_profile_state(conn, user_id)

    def _ensure_device_calibration(
        self,
        conn: object,
        *,
        user_id: str,
        device_id: str,
        device_name: str,
        now: object,
    ) -> dict:
        calibration = self.profile_repository.get_device_calibration(conn, user_id=user_id, device_id=device_id)
        if calibration is not None:
            return calibration
        created = self.profile_repository.create_device_calibration(
            conn,
            user_id=user_id,
            device_id=device_id,
            device_name=device_name,
            now=now,
        )
        if created is not None:
            return created
        return self.profile_repository.get_device_calibration(conn, user_id=user_id, device_id=device_id)

    def _reconcile_device(self, conn: object, *, user_id: str, device: DevicePayload, now: object) -> dict:
        existing = self.device_repository.get_by_client_key(conn, user_id, device.client_device_key)
        if existing is not None and existing["revoked_at"] is None:
            return self.device_repository.update_device_binding(
                conn,
                device_id=existing["device_id"],
                device_name=device.device_name,
                app_version=device.app_version,
                os_version=device.os_version,
                now=now,
            )

        active_device_count = self.device_repository.count_active_devices(conn, user_id)
        if active_device_count >= settings.device_limit:
            raise AppError(409, "device_limit_exceeded", "maximum active devices reached")

        if existing is not None:
            return self.device_repository.reactivate_device_binding(
                conn,
                device_id=existing["device_id"],
                device_name=device.device_name,
                app_version=device.app_version,
                os_version=device.os_version,
                now=now,
            )

        return self.device_repository.create_device_binding(
            conn,
            device_id=new_id("dev"),
            user_id=user_id,
            client_device_key=device.client_device_key,
            platform=device.platform,
            device_name=device.device_name,
            app_version=device.app_version,
            os_version=device.os_version,
            now=now,
        )
