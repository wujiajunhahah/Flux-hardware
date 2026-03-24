BEGIN;

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('active', 'disabled', 'deleted')),
    primary_email TEXT,
    timezone TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_identities (
    identity_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    provider_subject TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    UNIQUE (provider, provider_subject)
);

CREATE TABLE IF NOT EXISTS device_bindings (
    device_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    client_device_key TEXT NOT NULL,
    platform TEXT NOT NULL CHECK (platform IN ('ios', 'android', 'web')),
    device_name TEXT NOT NULL,
    app_version TEXT,
    os_version TEXT,
    last_seen_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_device_bindings_user_revoked_at
    ON device_bindings(user_id, revoked_at);

CREATE INDEX IF NOT EXISTS idx_device_bindings_user_active
    ON device_bindings(user_id)
    WHERE revoked_at IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_device_bindings_user_client_device_key
    ON device_bindings(user_id, client_device_key);

CREATE TABLE IF NOT EXISTS refresh_tokens (
    token_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    device_id TEXT NOT NULL REFERENCES device_bindings(device_id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user_device
    ON refresh_tokens(user_id, device_id);

CREATE TABLE IF NOT EXISTS model_releases (
    model_release_id TEXT PRIMARY KEY,
    model_kind TEXT NOT NULL CHECK (model_kind IN ('global_base_model', 'segment_model')),
    platform TEXT NOT NULL,
    channel TEXT NOT NULL CHECK (channel IN ('stable', 'beta', 'internal')),
    version TEXT NOT NULL,
    artifact_object_key TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    config_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL CHECK (status IN ('draft', 'published', 'retired')),
    published_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL,
    UNIQUE (platform, channel, model_kind, version)
);

CREATE INDEX IF NOT EXISTS idx_model_releases_platform_channel_status
    ON model_releases(platform, channel, status);

CREATE TABLE IF NOT EXISTS profile_states (
    profile_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL UNIQUE REFERENCES users(user_id) ON DELETE CASCADE,
    version INTEGER NOT NULL CHECK (version >= 1),
    calibration_offset NUMERIC(6, 2) NOT NULL,
    estimated_accuracy NUMERIC(5, 2) NOT NULL
        CHECK (estimated_accuracy >= 0 AND estimated_accuracy <= 100),
    training_count INTEGER NOT NULL CHECK (training_count >= 0),
    active_model_release_id TEXT REFERENCES model_releases(model_release_id),
    summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_profile_states_active_model_release_id
    ON profile_states(active_model_release_id);

CREATE TABLE IF NOT EXISTS device_calibrations (
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    device_id TEXT NOT NULL REFERENCES device_bindings(device_id) ON DELETE CASCADE,
    version INTEGER NOT NULL CHECK (version >= 1),
    device_name TEXT NOT NULL,
    sensor_profile_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    calibration_offset NUMERIC(6, 2) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (user_id, device_id)
);

CREATE INDEX IF NOT EXISTS idx_device_calibrations_device_id
    ON device_calibrations(device_id);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    device_id TEXT NOT NULL REFERENCES device_bindings(device_id) ON DELETE RESTRICT,
    idempotency_key TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending_upload', 'ready', 'failed', 'deleted')),
    source TEXT NOT NULL,
    title TEXT,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ NOT NULL,
    duration_sec INTEGER NOT NULL CHECK (duration_sec >= 0),
    snapshot_count INTEGER NOT NULL CHECK (snapshot_count >= 0),
    schema_version INTEGER NOT NULL CHECK (schema_version >= 1),
    content_type TEXT NOT NULL,
    blob_object_key TEXT,
    blob_size_bytes BIGINT CHECK (blob_size_bytes IS NULL OR blob_size_bytes >= 0),
    blob_sha256 TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    UNIQUE (user_id, idempotency_key),
    CHECK (ended_at >= started_at)
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_started_at_desc
    ON sessions(user_id, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_sessions_device_started_at_desc
    ON sessions(device_id, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_sessions_status_updated_at_desc
    ON sessions(status, updated_at DESC);

CREATE TABLE IF NOT EXISTS feedback_events (
    feedback_event_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    device_id TEXT NOT NULL REFERENCES device_bindings(device_id) ON DELETE RESTRICT,
    session_id TEXT REFERENCES sessions(session_id) ON DELETE SET NULL,
    idempotency_key TEXT NOT NULL,
    predicted_stamina INTEGER NOT NULL CHECK (predicted_stamina >= 0 AND predicted_stamina <= 100),
    actual_stamina INTEGER NOT NULL CHECK (actual_stamina >= 0 AND actual_stamina <= 100),
    label TEXT NOT NULL CHECK (label IN ('fatigued', 'alert')),
    kss INTEGER CHECK (kss IS NULL OR (kss >= 1 AND kss <= 9)),
    note TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    UNIQUE (user_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_feedback_events_user_created_at_desc
    ON feedback_events(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_feedback_events_session_id
    ON feedback_events(session_id);

CREATE INDEX IF NOT EXISTS idx_feedback_events_label_created_at_desc
    ON feedback_events(label, created_at DESC);

COMMIT;
