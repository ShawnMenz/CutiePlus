-- The cooperation database, version 1. Times are UTC text in one fixed format
-- (cutieplus/cooperation/clock.py), so comparing them as strings compares the times.
-- Ids are UUIDs, except audit_events, numbered in order.

CREATE TABLE users (
    id TEXT PRIMARY KEY,
    login_name TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    contact TEXT,
    role TEXT NOT NULL CHECK (role IN ('admin', 'supervisor', 'annotator')),
    status TEXT NOT NULL CHECK (status IN ('pending', 'active', 'suspended', 'archived')),
    password_hash TEXT,
    credential_setup_required INTEGER NOT NULL DEFAULT 1
        CHECK (credential_setup_required IN (0, 1)),
    setup_token_hash TEXT UNIQUE,
    setup_token_expires_at TEXT,
    authz_version INTEGER NOT NULL DEFAULT 1,
    row_version INTEGER NOT NULL DEFAULT 1,
    created_by TEXT REFERENCES users(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_login_at TEXT,
    suspended_at TEXT,
    archived_at TEXT,
    CHECK ((setup_token_hash IS NULL) = (setup_token_expires_at IS NULL))
);

CREATE TABLE ssh_keys (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    public_key TEXT NOT NULL,
    fingerprint_sha256 TEXT NOT NULL UNIQUE,
    device_label TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'active', 'revoked')),
    created_by TEXT REFERENCES users(id),
    created_at TEXT NOT NULL,
    expires_at TEXT,
    revoked_by TEXT REFERENCES users(id),
    revoked_at TEXT,
    last_authenticated_at TEXT
);
CREATE INDEX ssh_keys_user_status ON ssh_keys(user_id, status);

CREATE TABLE desktop_slots (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL UNIQUE REFERENCES users(id),
    display_number INTEGER NOT NULL UNIQUE,
    vnc_port INTEGER NOT NULL UNIQUE,
    workspace_path TEXT NOT NULL,
    service_name TEXT NOT NULL UNIQUE,
    vnc_password_file_reference TEXT,
    desired_state TEXT NOT NULL CHECK (desired_state IN ('running', 'stopped')),
    observed_state TEXT NOT NULL
        CHECK (observed_state IN ('starting', 'running', 'stopping', 'stopped', 'error')),
    runtime_generation INTEGER NOT NULL DEFAULT 0,
    process_identity TEXT,  -- JSON: {pid, start_ticks, boot_id}
    last_heartbeat_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_error TEXT
);

CREATE TABLE app_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    token_hash TEXT NOT NULL UNIQUE,
    desktop_slot_id TEXT REFERENCES desktop_slots(id) ON DELETE SET NULL,  -- NULL: the console
    issued_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    revoked_at TEXT,
    authz_version_at_issue INTEGER NOT NULL,
    last_reauthenticated_at TEXT
);
CREATE INDEX app_sessions_user_revoked ON app_sessions(user_id, revoked_at);

CREATE TABLE ssh_connections (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    ssh_key_id TEXT NOT NULL REFERENCES ssh_keys(id),
    desktop_slot_id TEXT REFERENCES desktop_slots(id) ON DELETE SET NULL,
    source_address TEXT NOT NULL,
    source_port INTEGER NOT NULL,
    ssh_entrypoint TEXT NOT NULL,
    connected_at TEXT NOT NULL,
    disconnected_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('active', 'disconnecting', 'closed')),
    server_process_identity TEXT NOT NULL,  -- JSON: {pid, start_ticks, boot_id}
    disconnect_reason TEXT
);
CREATE INDEX ssh_connections_key_status ON ssh_connections(ssh_key_id, status);

CREATE TABLE gpu_jobs (
    id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL REFERENCES users(id),
    submitted_by_app_session_id TEXT NOT NULL REFERENCES app_sessions(id),
    desktop_slot_id TEXT REFERENCES desktop_slots(id) ON DELETE SET NULL,
    operation_type TEXT NOT NULL CHECK (operation_type IN ('propagate', 'click', 'click_undo')),
    workspace_id TEXT NOT NULL,  -- the task's work folder
    input_reference TEXT NOT NULL,  -- the immutable snapshot
    input_revision TEXT NOT NULL,  -- the snapshot's SHA-256
    annotation_or_anchor_revision INTEGER NOT NULL,  -- the task's anchors revision at submission
    model_identifier TEXT NOT NULL,
    model_version TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    output_reference TEXT,  -- the job's staging folder
    result_manifest_reference TEXT,
    result_manifest_sha256 TEXT,
    -- whether the window applied the result to the task (none: no result)
    result_state TEXT NOT NULL DEFAULT 'none'
        CHECK (result_state IN ('none', 'pending', 'applied', 'held_for_review', 'discarded')),
    applied_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'cancel_requested', 'succeeded',
                                           'failed', 'cancelled', 'interrupted')),
    queue_sequence INTEGER NOT NULL UNIQUE,
    idempotency_key TEXT NOT NULL,
    execution_token TEXT,
    worker_instance_id TEXT,
    worker_generation INTEGER,
    progress REAL NOT NULL DEFAULT 0,
    submitted_at TEXT NOT NULL,
    started_at TEXT,
    heartbeat_at TEXT,
    finished_at TEXT,
    cancel_requested_at TEXT,
    cancelled_by TEXT REFERENCES users(id),
    retry_of_job_id TEXT REFERENCES gpu_jobs(id),
    checkpoint_reference TEXT,
    error_code TEXT,
    error_summary TEXT,
    UNIQUE (owner_user_id, idempotency_key)
);
CREATE INDEX gpu_jobs_status_sequence ON gpu_jobs(status, queue_sequence);
CREATE INDEX gpu_jobs_owner_submitted ON gpu_jobs(owner_user_id, submitted_at);
CREATE INDEX gpu_jobs_workspace ON gpu_jobs(workspace_id, result_state);

CREATE TABLE gpu_state (
    gpu_id TEXT PRIMARY KEY,
    worker_instance_id TEXT,
    worker_generation INTEGER NOT NULL DEFAULT 0,
    worker_process_identity TEXT,  -- JSON: {pid, start_ticks, boot_id}
    active_job_id TEXT REFERENCES gpu_jobs(id),
    state TEXT NOT NULL CHECK (state IN ('starting', 'idle', 'busy', 'recovering', 'offline')),
    heartbeat_at TEXT,
    memory_used_mb INTEGER,
    memory_total_mb INTEGER,
    last_error TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE management_operations (
    id TEXT PRIMARY KEY,
    operation_type TEXT NOT NULL,
    actor_user_id TEXT REFERENCES users(id),  -- NULL: the system
    target_user_id TEXT REFERENCES users(id),
    target_resource_id TEXT,
    request_id TEXT UNIQUE,
    desired_revision INTEGER,
    validated_parameters_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL
        CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'superseded')),
    current_step TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    error_summary TEXT
);
CREATE INDEX management_operations_status_created ON management_operations(status, created_at);

CREATE TABLE policy_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    desired_revision INTEGER NOT NULL DEFAULT 0,
    applied_revision INTEGER NOT NULL DEFAULT 0,
    applied_snapshot_checksum TEXT,
    last_applied_at TEXT,
    last_apply_error TEXT
);
INSERT INTO policy_state (id) VALUES (1);

CREATE TABLE audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_user_id TEXT REFERENCES users(id),  -- NULL: the system, or an unauthenticated caller
    actor_source TEXT NOT NULL
        CHECK (actor_source IN ('authenticated_app', 'verified_sshd', 'system')),
    action TEXT NOT NULL,
    target_type TEXT,
    target_id TEXT,
    request_id TEXT,
    operation_id TEXT,
    before_state_json TEXT,
    after_state_json TEXT,
    result TEXT NOT NULL CHECK (result IN ('succeeded', 'failed', 'refused')),
    error_code TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX audit_events_target ON audit_events(target_type, target_id, created_at);
CREATE INDEX audit_events_actor ON audit_events(actor_user_id, created_at);

CREATE TABLE system_settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_by TEXT REFERENCES users(id),
    updated_at TEXT NOT NULL
);
