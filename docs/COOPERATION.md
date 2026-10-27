# Cooperation: several people, one GPU

CutiePlus can let several people label at once on one Linux computer with one GPU: the owner at the
monitor, and collaborators who reach a remote desktop of their own over SSH and VNC. Everyone runs
as the same Linux account, but signs in to CutiePlus with an identity of their own, with one of
three roles, and every model runs in one shared GPU worker.

| Role | May |
|---|---|
| Annotator | label; their own GPU jobs (see, cancel, retry); their own password and sign-out |
| Supervisor | the same, and everything about annotators: add, edit, suspend, reactivate, remove, reset their credential, their SSH keys, desktops, sessions and jobs |
| Admin | everything about annotators and supervisors, make or revoke supervisors, every job, the settings, the whole audit log |

The first admin is the **owner**, created by a maintenance command on the computer itself. Nothing
in the application can suspend, remove, demote or reset the owner; only the maintenance commands
below can recover or hand over the ownership. Being an admin gives no access to the operating
system.

## How it fits together

```
your login at the monitor ── Cutie window ─┐
cutieplus-desktop@N (one per collaborator) ─┤ Xtigervnc on 127.0.0.1:5900+N, openbox, Cutie window
                                            ▼ /run/cutieplus/app.sock (sign-in required)
cutieplus-service   the cutieplus account; /opt/cutieplus code; /var/lib/cutieplus database
      │ worker.sock                     │ helper.sock (the cutieplus account only)
      ▼                                 ▼
cutieplus-gpu-worker                cutieplus-helper (root): the SSH policy file, VNC passwords,
  the only process using the GPU      starting and stopping desktops, cutting SSH connections
cutieplus-sshd (port 2222): collaborators' keys only, each forwarding to its own desktop
```

- **Sign-in.** Every Cutie window signs in, at every start; the session lives in that window only.
  A desktop's windows sign in as that desktop's user only; at the monitor anyone may. A change of
  someone's role, a suspension or a credential reset ends their sessions at once: they sign in
  again with what they may do now.
- **The GPU.** Windows never load a model. Clicks and propagations go to the one GPU worker:
  propagations wait in one queue (first come, first served, one at a time; no role goes first) and
  clicks run between the steps of the running one. A worker takes a lock on
  `/var/lib/cutieplus-gpu/gpu.lock` before it touches the GPU, so two never run. Other programs of
  the account can still use the GPU.
- **Results.** A propagation starts from a copy of its inputs taken when it was asked for. Its masks
  are written to the task when it ends, and only if the task did not change meanwhile; otherwise
  the result is held, and the window asks whether to apply it (anchors are never overwritten) or
  discard it. While a propagation waits for the GPU, its window stays editable.
- **SSH.** Collaborators' public keys are registered in Manage Annotators. Each becomes one line of
  `/etc/cutieplus/ssh/authorized_keys`, which the helper writes and reads back: forwarding to that
  user's own VNC port only, no shell, no SFTP, expiring if the key has a date. Port 22, your own,
  keeps its configuration.
- **What the database holds.** Users, password hashes (Argon2id), token hashes, keys, desktops,
  connections, GPU jobs and the audit log, in SQLite (`/var/lib/cutieplus/cooperation.sqlite3`,
  readable by the service account only). Frames, masks and model weights stay files.

The shared Linux account limits how far one desktop is kept from another: any program of the
account can read every desktop's VNC credential, take the GPU lock, or change the repository's code
that the GPU worker and the desktops run. Sign-in, roles, SSH access and the database are
protected from it; the files of the account are not.

## Installing

Run as yourself (the account that owns the repository), with sudo:

```bash
sudo deploy/install.sh
```

It creates the `cutieplus` account, copies the cooperation code to `/opt/cutieplus` (root-owned,
with its own Python environment), writes `/etc/cutieplus/service.toml` and `client.toml`, the
annotation sshd's configuration, the systemd units, opens port 2222 in ufw, and starts the
service, the helper, the GPU worker and the annotation sshd. Run it again after updating the
repository. From then on every Cutie window on this computer signs in.

Then create the owner and set your password:

```bash
sudo cutieplus-admin bootstrap-owner --login <your login> --name "<your name>"
uv run python interactive_demo.py      # at the monitor: "First time?", the token, a password
```

`sudo deploy/install.sh --disable` stops it all and removes `client.toml`: windows started
afterwards run the models themselves again, as before. The database and settings are kept.

Once everything works, `sudo deploy/install.sh --port22` makes port 22 read keys from a root-owned
file (`/etc/ssh/authorized_keys.d/<account>`) instead of `~/.ssh/authorized_keys`, which any
window of the account could change. Your password sign-in is not affected.

Settings that may change while it runs (session lengths, the pending-job limit, backups) are in
Manage Annotators → Settings. The rest is `/etc/cutieplus/service.toml`; after changing it, restart
the services (`sudo systemctl restart cutieplus-service cutieplus-helper cutieplus-gpu-worker`) and,
for the SSH port or display range, run the installer again.

## Adding a collaborator

1. In the task browser, **Manage Annotators → Add Annotator** (an admin picks the role). The
   collaborator gets a desktop (`:2`, `:3`, … up to `:20`) and a card is shown once: their setup
   token, their desktop's VNC password, and the SSH command. Hand it on privately.
2. They send you their **public** key (the one line of `id_ed25519.pub`; `ssh-keygen -t ed25519`
   makes one). Select them, **Add key…**, paste it, name the device. Private keys are refused.
3. On their computer:

   ```bash
   ssh -N -L 5901:localhost:5902 -p 2222 <account>@<this computer>
   ```

   then a VNC viewer to `localhost:5901` with the VNC password. In the Cutie window there:
   **First time?**, their login name, the setup token, and a password of their own.

If they connect from outside your network, forward port 2222 of your router to this computer.
A desktop keeps running when they disconnect; they come back to the same one.

## Appointing a supervisor

An admin selects the annotator in Manage Annotators and clicks **Make Supervisor** (it asks for the
admin's password again if they signed in more than 10 minutes ago). The user is signed out and,
after signing in again, sees Manage Annotators with the annotators in it. **Revoke Supervisor**
makes them an annotator again; their desktop and work stay.

## Revoking a key, disconnecting

- **Revoke** (the key's row): the key stops working once the policy file is rewritten, and its
  open connections are cut; their other keys keep working. A revoked key never comes back: they
  make a new one.
- **Disconnect** (a connection's row): ends that SSH connection only; they may connect again.

The Operations card shows each change as *Applying* until the system has done it, and *Failed* with
the reason while the service keeps trying (for example, if the helper is down).

## Suspending and removing

- **Suspend**: they are signed out everywhere, their SSH access stops and their connections are
  cut, their desktop stops (its window saves and closes), and their waiting GPU jobs are cancelled.
  **Reactivate** brings access and desktop back.
- **Remove**: the same, for good: their keys are revoked, they are archived, and their desktop is
  freed once it has stopped. Their annotation files, jobs and audit records are kept.
- **Reset credential**: their password stops working; a new setup token is shown once.

## The GPU

The **GPU queue** button (every role) shows the worker, the running and waiting propagations
(owners as far as you may see them) and your own, with Cancel and Retry.

- Cancelling a waiting propagation is immediate; a running one stops after its current step, once
  the worker has freed it. The frames it made are kept, as with Pause.
- If the worker crashes, its running job is *interrupted* with the frames it made; nothing is
  retried by itself. **Retry** queues a new job from the same inputs.
- If the worker stops answering, the service restarts its unit (systemd ends every process of the
  old one before the new one starts). By hand:
  `sudo systemctl restart cutieplus-gpu-worker` and `journalctl -u cutieplus-gpu-worker`.
- Out of GPU memory, a job fails (*oom*) and the queue goes on. Close other programs using the GPU
  (`nvidia-smi`), then Retry.
- `sudo cutieplus-admin status` sums up users, the SSH policy, the worker and the jobs.

## Backups and restoring

The service backs the database up every 24 hours (Settings: `backup_interval_hours`, `backup_keep`)
into `/var/lib/cutieplus/backups/`; `sudo cutieplus-admin backup` makes one now. To restore:

```bash
sudo cutieplus-admin restore /var/lib/cutieplus/backups/cooperation-<time>.sqlite3
```

It checks the backup, stops the service, puts the backup in place, and starts the service, which
then reconciles on the safe side: every session ends, jobs that were running are interrupted, a key
revoked after the backup stays revoked, and the SSH policy is written again from the backup (keys
added after the backup must be added again). It lists the users: changes made after the backup
(a suspension, a new user) are not in it, so check them.

## Recovering the owner

```bash
sudo cutieplus-admin owner-reset                    # a new setup token for the owner
sudo cutieplus-admin owner-transfer --login <user>  # an active user becomes the owner and admin;
                                                    # the previous owner becomes a supervisor
```

## Logs

`journalctl -u cutieplus-service`, `-u cutieplus-helper`, `-u cutieplus-gpu-worker`,
`-u cutieplus-sshd`, `-u 'cutieplus-desktop@*'`. The audit log (Manage Annotators → Audit, admins)
records sign-ins, every management action with its actor, SSH connections as sshd's journal reports
them, and the end of every GPU job.
