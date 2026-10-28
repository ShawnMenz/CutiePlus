-- A user deleted for good (users.delete_user) leaves the audit log as it was: the entries of what
-- they did keep their login name here, since actor_user_id then names nobody.
ALTER TABLE audit_events ADD COLUMN actor_login_name TEXT;
