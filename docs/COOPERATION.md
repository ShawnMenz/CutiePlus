# Multi-annotator deployment

The optional cooperation service lets several annotators use one Linux host and one shared GPU. The operator can work at the host's display, while remote annotators connect to individual desktops through SSH and VNC. Each person signs in with a CutiePlus account. All remote desktops and the GPU worker run under the same Linux desktop account.

Complete the [Quick Start](../README.md#quick-start) before deploying this service. For dataset preparation and annotation, see the [workflow guide](WORKFLOW.md).

## Install on the GPU host

### Requirements

- A systemd-based Linux host with a working NVIDIA GPU and a graphical desktop for local operation.
- A checkout with its `.venv` and model weights prepared using the README instructions. The worker and remote desktops use this checkout directly; keep it at a stable path.
- The host's `/usr/bin/python3` must provide `tomllib` (Python 3.11 or later). The installer uses it for the helper and a separate service environment, independently of the project's Python 3.12 environment.
- `sudo`, `uv`, and network access to install the service dependencies.
- An SSH server reachable by collaborators on port 22. Remote annotators need an SSH client, a public key, and a VNC viewer.

On Ubuntu, install the remote desktop dependencies:

```bash
sudo apt install tigervnc-standalone-server tigervnc-tools openbox dbus-x11 openssh-server
```

### Install and check the services

Run the installer from the repository as the desktop account that owns the checkout, using `sudo`. It requires `SUDO_USER` and refuses a direct root login.

```bash
sudo deploy/install.sh
systemctl status cutieplus-service cutieplus-helper cutieplus-gpu-worker cutieplus-sshd
```

The installer creates two system accounts, writes systemd units and SSH configuration, and starts four services:

| Component | Purpose |
| --- | --- |
| `cutieplus-service` | Application accounts, permissions, database, management operations, and GPU queue; runs as the protected `cutieplus` account. |
| `cutieplus-helper` | A root service with fixed operations for SSH policy, VNC credentials, desktops, and connections. |
| `cutieplus-gpu-worker` | Model inference under the desktop account, using the checkout's `.venv` and weights. |
| `cutieplus-sshd` | An annotation SSH server listening on the host's loopback interface, port 2222 by default. |

The service and helper use a root-owned release in `/opt/cutieplus`, with a separate environment in `/opt/cutieplus/venv`. The installer also adds `/etc/ssh/sshd_config.d/20-cutieplus-jump.conf` to the host's existing SSH server: collaborators enter as `cutieplus-jump` and can forward only to the annotation SSH server. The regular installation changes SSH policy for that jump account; the optional host-wide key policy is described below.

If a service fails, inspect its journal before continuing:

```bash
sudo journalctl -u cutieplus-service -u cutieplus-helper -u cutieplus-gpu-worker -u cutieplus-sshd -n 100 --no-pager
```

The installer normally uses the host's IPv4 address on its default route for connection instructions. If collaborators need another reachable address or DNS name, specify it explicitly:

```bash
sudo deploy/install.sh --ssh-host gpu.example.org
```

Replace `gpu.example.org` with your host's address. This also runs the normal installation/update steps. Do not expose port 2222 or the VNC ports externally; collaborators reach them through the SSH tunnel on port 22.

### Create the owner

Choose a CutiePlus login and display name; these are application credentials, separate from Linux accounts. For example:

```bash
sudo cutieplus-admin bootstrap-owner --login alex --name "Alex Chen"
uv run python interactive_demo.py
```

At the host's display, select **First time, or after a reset? Use your setup token**, enter the login and the printed token, and set a password. Passwords must contain 10–256 characters and cannot equal the login name. Setup tokens are shown once and expire after 72 hours by default.

The owner is the application's admin. This version permits creating annotators and supervisors; ownership transfer is the maintenance command for changing the admin. The owner cannot be suspended, removed, deleted, demoted, or have their credential reset through the application.

Installing `/etc/cutieplus/client.toml` enables cooperation mode for newly started windows on this host. Each window requires sign-in and uses the GPU service. A managed remote desktop accepts only its assigned user; a window at the host's display accepts any active application user. Plain-workspace launches (`--images`, `--video`, or `--workspace` without `--tasks`) are refused in cooperation mode; use the task browser.

## Connect an annotator

### Prepare the account

1. Open **Manage Annotators** in the task browser. An admin selects **Add user** and chooses Annotator or Supervisor; a supervisor selects **Add Annotator**.
2. Enter the login and display name. Logins have 2–32 characters: a letter followed by letters, digits, `.`, `_`, or `-`; they are normalized to lowercase. Some system and role names are reserved for users created in the application.
3. Copy the setup token, VNC password, and connection instructions from the resulting card and deliver them privately. New users receive a desktop automatically, using displays `:2` through `:20` by default. Wait for its management operation to finish before connecting.
4. Obtain the annotator's SSH **public** key, such as the contents of `~/.ssh/id_ed25519.pub`. Select the user, choose **Add key…**, paste the key, and name the device. The application rejects private keys. An annotator can generate an Ed25519 key with `ssh-keygen -t ed25519` if needed.

**Connection info…** displays the login, SSH command, and VNC address later. A setup token or VNC password is displayed only when generated. **New setup token** and **New VNC password** replace the old values; generating a VNC password restarts the desktop. The database keeps hashes of application credentials and tokens. VNC authentication uses a credential file on the host.

### Open the remote desktop

On the annotator's computer, run the exact SSH command shown in **Connection info…**. For a desktop on display `:2`, it has this form:

```bash
ssh -N -o ExitOnForwardFailure=yes -J cutieplus-jump@gpu.example.org -p 2222 \
    -L 5902:localhost:5902 desktop_account@localhost
```

Use the generated command's host, Linux account, and port values. The CutiePlus login is entered in the application, not substituted for `desktop_account`. The public key must be available to the SSH client for both hops. Verify the host keys with the operator when connecting for the first time.

Keep the SSH process running. In a VNC viewer, connect to the address from the card (`localhost:5902` in this example) and enter the VNC password. In the CutiePlus window, use the setup-token option to create the application password, then annotate normally.

SSH forwarding and VNC address syntax depends on the viewer: some viewers use `localhost::5902` to select TCP port 5902 explicitly. Use the supplied port rather than treating 5902 as a display number.

Disconnecting the SSH tunnel leaves the remote desktop running, so the annotator can reconnect to the same session. Closing the application on a remote desktop starts it again after a short delay. Remote desktops provide CutiePlus with a restricted Openbox configuration; they do not provide a general-purpose terminal or desktop menu.

## Manage users and work

### Roles and access

| Role | Permissions |
| --- | --- |
| Annotator | Annotate; inspect, cancel, and retry their own GPU jobs; change their password and sign out. |
| Supervisor | Annotator permissions, plus account, key, desktop, session, and job management for annotators. |
| Admin | Manage annotators and supervisors; appoint or revoke supervisors; permanently delete users; manage all jobs, settings, and the audit log. |

An admin can select **Make Supervisor** or **Revoke Supervisor**. A role change ends that user's application sessions; their desktop and annotation files remain. The service checks permissions against the current target role on every request. Changing a role, removing a user, or changing settings requires the manager's password again when the last authentication is older than 10 minutes by default. Permanent deletion asks for it every time.

| Action | Effect |
| --- | --- |
| **Revoke** a key | Removes that key from the published SSH policy and disconnects its tracked connections. Other keys remain usable. A revoked key cannot be reactivated. |
| **Disconnect** a connection | Ends that SSH connection; the key can reconnect. |
| **Suspend** | Ends application sessions, disables SSH access, stops the desktop, cancels queued jobs, and requests cancellation of a running job. **Reactivate** restores access and the desktop. |
| **Remove** | Archives the user, revokes their keys, ends access, and releases the desktop after it stops. Annotation files, job records, and audit history remain. |
| **Reset credential** | Invalidates the password and application sessions and issues a new setup token. |
| **Delete permanently…** | Admin-only deletion after access has ended and the desktop is released. Removes the account, keys, sessions, and job records. Annotation files and audit history remain; the login can be reused. Older backups may still contain the deleted records. |

Changes to desktops and SSH access are asynchronous. The **Operations** card shows **Applying** until the system change completes and **Failed** with a reason while the reconciler retries. Check this status when the GUI has accepted a request but access has not changed yet.

### GPU queue and results

**GPU queue** shows the worker and propagation jobs that the signed-in user may manage. Propagations run in submission order, one at a time, with no role priority. Clicks are scheduled between propagation steps and shared between users. The default limit of three pending propagations per user includes queued, running, and cancelling jobs.

A propagation records its submitted inputs and stages its output separately. Its window applies the result only if the task has the expected revision; otherwise it offers the result for review. Applying a held result preserves anchors. A submitted job can finish while its window is closed; its owner's window checks for pending results when that task is opened again.

- **Cancel** stops a queued job immediately. A running job stops after the current step and releases its GPU state; completed frames remain available as a partial result.
- A worker failure interrupts its running job. Jobs are not automatically retried. **Retry** is available for failed, interrupted, or cancelled propagations and creates a new job from the original submitted inputs.
- An out-of-memory failure is reported as `oom`. Check other GPU processes with `nvidia-smi`, free memory as appropriate, and retry.

The worker holds `/var/lib/cutieplus-gpu/gpu.lock` before initializing its GPU work. This prevents a second cooperating worker from running; it does not prevent unrelated programs from using the GPU.

## Configuration and updates

| Path | Contents |
| --- | --- |
| `/etc/cutieplus/service.toml` | Paths, SSH address, desktop range, and service configuration; root-owned. |
| `/etc/cutieplus/client.toml` | Enables cooperation mode and supplies the application socket. |
| `/var/lib/cutieplus/` | Service-owned database, job input snapshots, result manifests, and database backups; mode `0700`. |
| `<checkout>/workspace/.cutie_jobs/` | Staged GPU output files by default. |
| `/etc/cutieplus/ssh/` | Published SSH key policies, written by the helper. |
| `/var/lib/cutieplus-desktops/` | Remote desktop environment and VNC credential files. |

The default input is `<checkout>/workspace/cutie_tasks`. GPU requests require task workspace and frame paths inside the configured `workspace_root`; setting `--tasks` to an arbitrary external path does not change that boundary. Configure and prepare the workspace consistently before using another location.

Admins can change session timeouts, setup-token lifetime, pending-job limit, and backup settings in **Manage Annotators → Settings**. Defaults are 12 hours of session inactivity, a seven-day maximum session lifetime, a 72-hour setup-token lifetime, a 24-hour backup interval, and 14 retained database backups.

Edit `/etc/cutieplus/service.toml` as root for host configuration, then restart the service, helper, and worker. Re-run the installer when changing SSH or desktop-range configuration so it regenerates the SSH configuration and units. The supplied desktop script uses VNC ports `5900 + display`; retain that mapping.

After updating the checkout and syncing its dependencies, close application windows and re-run `sudo deploy/install.sh` as the desktop account. This refreshes the protected service release and restarts its services; schedule it when annotators can stop work because running inference is interrupted. The existing `service.toml` and database are retained.

### Optional host-wide SSH key policy

`sudo deploy/install.sh --port22` is a separate operation that changes the host's regular SSH server to read public keys from `/etc/ssh/authorized_keys.d/%u` through `/etc/ssh/sshd_config.d/10-cutieplus-keys.conf`. The installer copies only the desktop account's existing `~/.ssh/authorized_keys`, and only if its destination does not exist. Other accounts' public-key access can be affected because their keys are not copied.

Before using this option, provision the root-owned key files needed by other accounts and check the host's effective SSH configuration. Keep an administrative session open while testing access. The installer does not change password authentication. This option addresses the shared desktop account's ability to edit its own `~/.ssh/authorized_keys`; it is not required to run the annotation service.

### Disable cooperation mode

```bash
sudo deploy/install.sh --disable
```

This stops and disables the services, stops remote desktops, removes `client.toml`, and removes the jump-account SSH drop-in. Newly started windows then use local inference. The database, settings, accounts, and stored key files remain. The optional host-wide `10-cutieplus-keys.conf` policy is not removed by this command.

## Backup and recovery

### Back up and restore the database

Automatic backups are written to `/var/lib/cutieplus/backups/` by default. Create one immediately with:

```bash
sudo cutieplus-admin backup
```

The command prints its path. These backups contain the cooperation database, including user and audit records. They do not include task images, annotations, model weights, the checkout, or job snapshot and result files. Back up those separately if needed.

Restore a database backup with:

```bash
sudo cutieplus-admin restore /var/lib/cutieplus/backups/cooperation-<time>.sqlite3
```

Replace the placeholder with an existing backup path readable by the `cutieplus` service account. The command stops the service, checks and restores the database as that account, and starts the service again. Application sessions end; jobs recorded as running are interrupted. The SSH policy and desired desktop states are reconciled from the restored database.

When the current SSH policy file is readable, active keys absent from that policy are revoked in the restored database, preserving later revocations. Keys added after the backup must be added again. Other changes after the backup, including suspensions and newly created users, are absent: review the users printed by the command and verify access before resuming work.

### Recover or transfer ownership

```bash
sudo cutieplus-admin owner-reset
sudo cutieplus-admin owner-transfer --login jordan
```

`owner-reset` clears the owner's password, ends their sessions, and prints a new setup token. `owner-transfer` requires an existing active user; it makes that user the owner and admin and demotes the previous owner to supervisor. Both users sign in again.

### Reset all cooperation records

```bash
sudo cutieplus-admin reset
```

After the operator types `RESET`, this command stops desktops and relevant services and clears all cooperation records, including the owner, keys, jobs, and audit log. It keeps a checked database copy in the backup directory and leaves annotation files untouched. Existing database backups still contain old records. After a reset, create the owner again with `bootstrap-owner`.

## Troubleshooting

| Symptom | Checks |
| --- | --- |
| Application cannot reach the service | Check `cutieplus-service` and its journal. Installing `client.toml` enables service mode even when the service is stopped. |
| SSH authentication fails | Confirm the key was added to the intended active user and its operation completed. Use the generated connection command; inspect `cutieplus-sshd` and the host's SSH logs. |
| SSH reports a local port already in use | Close the earlier tunnel or choose an unused local port and use the same local port in the VNC viewer. Keep the remote destination port unchanged. |
| VNC rejects the password | Use the most recently generated VNC password and check the desktop's operation status. A new password restarts the desktop. |
| Remote desktop does not start | Inspect `journalctl -u cutieplus-desktop@2` for display `:2`, substituting the assigned display. Check the checkout and `.venv` paths. |
| No desktop is available | All configured slots may be assigned. Review users and released desktops; removed users release their slot after stopping. |
| Job is stuck or the worker is unavailable | Check **GPU queue**, `sudo cutieplus-admin status`, and `journalctl -u cutieplus-gpu-worker`. If required, restart it with `sudo systemctl restart cutieplus-gpu-worker`; a running job is interrupted. |
| Request says a frame or workspace is outside the workspace | Compare the task and frame paths with `workspace_root` in `service.toml`. |
| Management operation remains failed | Read its reported reason and the helper/service logs. The reconciler retries; restarting the application alone does not repair a failed system operation. |

Admins can read application audit records in **Manage Annotators → Audit**. It records authentication, management actions, SSH connections observed from the annotation SSH journal, and job completion. System logs are available through `journalctl` for each service.

## Trust boundary

Application users have distinct identities and roles, but remote desktops share the Linux desktop account. This deployment is intended for trusted collaborators sharing a dataset, rather than operating-system isolation between users. A process under the shared account can access that account's files and VNC credentials or modify the checkout used by the worker and desktops. The service database and protected service release use a separate account and root-owned code; application admin rights do not grant Linux administrative privileges.
