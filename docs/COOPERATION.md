# Cooperation: several people, one GPU

CutiePlus can let several people label at once on one Linux computer with one GPU: the owner at the
monitor, and collaborators who reach a remote desktop of their own over SSH and VNC. Everyone runs
as the same Linux account, but signs in to CutiePlus with an identity of their own, with one of
three roles, and every model runs in one shared GPU worker.

| Role | May |
|---|---|
| Annotator | label; their own GPU jobs (see, cancel, retry); their own password and sign-out |
| Supervisor | the same, and everything about annotators: add, edit, suspend, reactivate, remove, reset their credential, their SSH keys, desktops, sessions and jobs |
| Admin | everything about annotators and supervisors, make or revoke supervisors, delete users for good, every job, the settings, the whole audit log |

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
port 22 (your sshd): collaborators sign in as cutieplus-jump, which may only hop on to
  ▼
cutieplus-sshd (port 2222, this computer only): each key forwards to its own desktop only
```

- **Sign-in.** Every Cutie window signs in, at every start; the session lives in that window only.
  A desktop's windows sign in as that desktop's user only; at the monitor anyone may. A change of
  someone's role, a suspension or a credential reset ends their sessions at once: they sign in
  again with what they may do now. Quit on the sign-in page ends Cutie; on a remote desktop it
  opens again a moment later, since the desktop runs nothing else.
- **A remote desktop runs Cutie only.** Its openbox gets the release's configuration
  (`deploy/openbox/`): no menu, no key binding, nothing that starts a program. Openbox's own
  defaults would offer a terminal, a shell as the desktop account, from a right click.
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
  user's own VNC port only, no shell, no SFTP, expiring if the key has a date. The annotation
  sshd listens on this computer only: collaborators come in through port 22, since many networks'
  firewalls (campus networks often) pass SSH on port 22 alone (on another port they let the
  connection open, then drop everything). There they sign in as `cutieplus-jump`, an account of
  its own (no shell, no password), with the same keys (`/etc/cutieplus/ssh/jump_keys`), and may
  open a channel to the annotation sshd and nothing else: no shell, no command, no other port, no
  Unix socket, nothing listening (`ssh -J`). Port 22's sshd holds that account to it through one
  drop-in, `/etc/ssh/sshd_config.d/20-cutieplus-jump.conf`; nobody else's settings change, yours
  included.
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
annotation sshd's configuration, the systemd units, port 22's drop-in (checked by sshd before
port 22 reloads), and starts the service, the helper, the GPU worker and the annotation sshd. Run it again after updating the
repository. From then on every Cutie window on this computer signs in.

Collaborators' SSH command names this computer by its IP address on the default route, never by
its host name (which resolves on this computer only). To name it otherwise, e.g. by a DNS name,
`sudo deploy/install.sh --ssh-host <name or address>` sets `ssh_host` in `service.toml` (and
updates as usual).

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
   Later, **Connection info…** (beside the selected user's name) shows their login name, the SSH
   command and the VNC address to copy at any time. The setup token and the VNC password are
   never kept, so the card makes new ones on request (**New setup token**, **New VNC password**:
   the old ones stop working).
2. They send you their **public** key (the one line of `id_ed25519.pub`; `ssh-keygen -t ed25519`
   makes one). Select them, **Add key…**, paste it, name the device. Private keys are refused.
3. On their computer:

   ```bash
   ssh -N -o ExitOnForwardFailure=yes -J cutieplus-jump@<this computer's address> -p 2222 \
       -L 5902:localhost:5902 <account>@localhost
   ```

   `-J` signs in on port 22 as `cutieplus-jump` with their key and hops to the annotation sshd
   (`localhost`, port 2222, on this computer), which forwards their own desktop. The first time,
   ssh asks them to accept two host keys: this computer's, then the annotation sshd's
   (`localhost`).

   Then a VNC viewer to `localhost:5902` with the VNC password: their computer forwards the
   desktop's own port (5902 for `:2`, 5903 for `:3`, …), so a VNC server of theirs (5900, 5901) or
   another desktop's tunnel does not collide. If that port is in use there anyway, ssh says so and
   ends: an earlier tunnel may still run (`lsof -nP -iTCP:5902 -sTCP:LISTEN` on a Mac or Linux,
   `netstat -ano | findstr :5902` on Windows). In the Cutie window there: **First time, or after
   a reset? Use your setup token**, their login name, the setup token, and a password of their
   own.

If they connect from outside your network, it is port 22 that must reach this computer (a
VPN, or a rule of the network's firewall); port 2222 stays closed to the outside.
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
- **Delete permanently…** (admins; never yourself or the owner): asks for your password every
  time, however recently you signed in. Their access ends at once, as with Remove; once their
  desktop has stopped and been released, their account, SSH keys, sessions and GPU job records are
  deleted and their login name may be given again. Their annotation files and the audit log are
  kept: the log still names them, as "<login> (deleted)" for what they did. Backups made before
  still hold them until they are rotated out.

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

## Starting over

```bash
sudo cutieplus-admin reset
```

After you type RESET, it deletes every record: every account (the owner's too), SSH key, desktop,
session, GPU job and the audit log. Collaborators' desktops stop and their connections end. A
checked copy of the database is kept in `/var/lib/cutieplus/backups/` (delete it, and the older
backups there, to remove the records for good). Annotation files are never touched. Then create
the owner again, as on the first day (`sudo cutieplus-admin bootstrap-owner …`).

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
