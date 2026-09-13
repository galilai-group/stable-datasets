"""Checkpoint callback that offloads checkpoints to Google Drive via rclone.

Why this exists: the ICLR grid is ~300+ cells x a few hundred MB of
optimizer-bearing checkpoints = several TB, which does not fit in OSCAR's
100 GB HOME or the 512 GB scratch soft quota. Drive is the cheap durable tier.

Design (mirrors the shape asked for in the sweep plan):

  1. One small ThreadPoolExecutor is created per callback instance and shares
     the callback's lifecycle (created in __init__, drained + shut down in
     teardown). Uploads are I/O-bound subprocess waits, so 2 workers saturate
     a Drive connection without stealing CPU from the dataloader.
  2. _save_checkpoint writes to local disk FIRST (via the parent
     ModelCheckpoint, so top-k/monitor/best bookkeeping is untouched), then
     hands the path to the pool and returns immediately. Training never blocks
     on the network.
  3. The worker uploads with rclone, VERIFIES the remote size matches local,
     and only then unlinks the local file.

Two invariants that matter more than the disk savings:

  * ``last.ckpt`` is NEVER deleted locally while the run is live. SLURM
    preemption/requeue resumes from it (benchmarks/run.py auto-resume), and a
    requeued job that has to re-download 300 MB from Drive before it can start
    -- or worse, finds the upload still in flight -- loses the run. It is
    uploaded every time (so Drive has a current copy if the node dies) but the
    local copy is only removed in teardown, after fit ends.
  * A local file is deleted ONLY after a verified upload. If rclone is missing,
    the token is expired, or the size check fails, the callback logs loudly and
    degrades to plain local checkpointing. A broken Drive must never cost us a
    4-day run's checkpoints.

Torn reads are not a concern: spt installs an atomic-checkpoint plugin
(stable_pretraining/utils/atomic_checkpoint.py) that writes to a sibling .tmp
and atomically renames, so rclone either reads the previous complete inode or
the new one -- never a half-written file.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint

log = logging.getLogger("gdrive_ckpt")


class GoogleDriveModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that mirrors checkpoints to Google Drive and frees local disk.

    Every stock ModelCheckpoint argument (monitor/mode/save_top_k/every_n_epochs/
    save_last/save_weights_only/filename/dirpath/...) is accepted and forwarded
    unchanged -- this subclasses rather than reimplements so "best" checkpoint
    selection stays exactly the upstream behaviour.

    Args:
        remote: rclone remote name, e.g. ``samibg-drive``. Combined with
            ``remote_dir`` to form ``<remote>:<remote_dir>/<run>/<file>.ckpt``.
        remote_dir: path within the remote, e.g. ``checkpoints``.
        folder_id: optional Drive folder id, recorded for provenance and used
            to build the human-facing URL in logs. rclone addresses the folder
            by path, so this is informational unless ``root_folder_id`` is set
            on the remote itself.
        rclone_binary: path to the rclone executable.
        rclone_config: optional explicit rclone.conf path.
        max_upload_workers: size of the shared upload pool.
        delete_local_after_upload: unlink local checkpoints once verified on
            Drive. ``last.ckpt`` is exempt until teardown (see module docstring).
        keep_best_local: also exempt the current ``best_model_path``. Turn this
            on if something downstream loads the best checkpoint from disk
            during the same job.
        upload_timeout_s: per-file rclone timeout.
        enabled: master switch; False makes this behave as a plain ModelCheckpoint.
    """

    def __init__(
        self,
        *,
        remote: str,
        remote_dir: str = "checkpoints",
        folder_id: str | None = None,
        rclone_binary: str = "rclone",
        rclone_config: str | None = None,
        max_upload_workers: int = 2,
        delete_local_after_upload: bool = True,
        keep_best_local: bool = False,
        upload_timeout_s: int = 3600,
        rclone_extra_args: list[str] | None = None,
        enabled: bool = True,
        **model_checkpoint_kwargs,
    ) -> None:
        super().__init__(**model_checkpoint_kwargs)
        self.remote = remote
        self.remote_dir = remote_dir.strip("/")
        self.folder_id = folder_id
        self.rclone_binary = rclone_binary
        self.rclone_config = rclone_config
        self.delete_local_after_upload = delete_local_after_upload
        self.keep_best_local = keep_best_local
        self.upload_timeout_s = upload_timeout_s
        self.rclone_extra_args = list(rclone_extra_args or [])
        self.enabled = enabled

        # Lifecycle-shared pool. Small on purpose: these workers only wait on an
        # rclone subprocess, and more parallel Drive streams mostly earns 429s.
        self._pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=max_upload_workers, thread_name_prefix="gdrive-ckpt")
            if enabled
            else None
        )
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        # Set once if Drive turns out to be unusable, so we stop trying (and,
        # critically, stop deleting local files) for the rest of the run.
        self._degraded = False

    # ---------------------------------------------------------------- helpers

    def _rclone_cmd(self, *args: str) -> list[str]:
        cmd = [self.rclone_binary]
        if self.rclone_config:
            cmd += ["--config", self.rclone_config]
        return cmd + list(args)

    def _is_last_ckpt(self, filepath: str) -> bool:
        """True for last.ckpt / last-v1.ckpt (mirrors ModelCheckpoint's own pattern)."""
        return re.match(rf"^{re.escape(self.CHECKPOINT_NAME_LAST)}(-(\d+))?$", Path(filepath).stem) is not None

    def _run_name(self, filepath: str) -> str:
        """Drive subfolder for this cell: the local checkpoint dir's basename,
        which run.py sets to ``{model}_{backbone}_{dataset}[_seed{N}]``."""
        return Path(filepath).parent.name or "run"

    def _remote_path(self, filepath: str) -> str:
        return f"{self.remote}:{self.remote_dir}/{self._run_name(filepath)}/{Path(filepath).name}"

    def _degrade(self, why: str) -> None:
        if not self._degraded:
            self._degraded = True
            log.error(
                "[gdrive] DISABLING Drive offload for the rest of this run: %s. "
                "Checkpoints stay on local disk -- watch the filesystem quota.",
                why,
            )

    # ------------------------------------------------------------ upload work

    def _upload_and_maybe_delete(self, filepath: str, may_delete: bool) -> bool:
        """Runs on a pool thread. Returns True iff the file is verified on Drive."""
        if self._degraded:
            return False
        remote = self._remote_path(filepath)
        try:
            # Racy by nature: top-k rotation can unlink this path between the
            # save that queued it and this thread waking up. That is a no-op,
            # NOT a Drive failure — must not trip the degrade path below.
            local_size = os.path.getsize(filepath)
        except FileNotFoundError:
            log.debug("[gdrive] %s vanished before upload (rotated out) — skipping", Path(filepath).name)
            return False
        try:
            proc = subprocess.run(
                self._rclone_cmd(
                    "copyto", filepath, remote,
                    "--retries", "5",
                    "--low-level-retries", "10",
                    "--drive-chunk-size", "64M",
                    *self.rclone_extra_args,
                ),
                capture_output=True, text=True, timeout=self.upload_timeout_s,
            )
            if proc.returncode != 0:
                err = (proc.stderr or "").strip().splitlines()
                tail = err[-1] if err else f"exit {proc.returncode}"
                if "invalid_grant" in (proc.stderr or "") or "token expired" in (proc.stderr or ""):
                    self._degrade(f"rclone auth failed ({tail}) -- run: rclone config reconnect {self.remote}:")
                else:
                    log.warning("[gdrive] upload failed for %s: %s", Path(filepath).name, tail)
                return False

            # Verify before we delete anything. rclone checks integrity itself,
            # but this is data we are about to destroy locally -- confirm the
            # byte count independently.
            chk = subprocess.run(
                self._rclone_cmd("size", "--json", remote),
                capture_output=True, text=True, timeout=120,
            )
            if chk.returncode != 0 or f'"bytes":{local_size}' not in chk.stdout.replace(" ", ""):
                log.warning(
                    "[gdrive] size mismatch after upload of %s (local=%d, remote=%s) -- keeping local copy",
                    Path(filepath).name, local_size, (chk.stdout or "").strip()[:120],
                )
                return False

            log.info("[gdrive] uploaded %s (%.1f MB) -> %s", Path(filepath).name, local_size / 1e6, remote)

            if may_delete and self.delete_local_after_upload:
                try:
                    os.remove(filepath)
                    log.info("[gdrive] freed local %s", filepath)
                except FileNotFoundError:
                    pass
                except OSError as e:
                    log.warning("[gdrive] could not unlink %s: %s", filepath, e)
            return True
        except subprocess.TimeoutExpired:
            log.warning("[gdrive] upload timed out after %ss for %s", self.upload_timeout_s, filepath)
            return False
        except FileNotFoundError:
            # Only reachable when the rclone executable itself is missing —
            # a missing *checkpoint* is handled above, before this try block.
            self._degrade(f"rclone binary not found at {self.rclone_binary!r}")
            return False
        except Exception as e:  # never let an upload thread kill training
            log.warning("[gdrive] unexpected upload error for %s: %s", filepath, e)
            return False

    def _submit(self, filepath: str, may_delete: bool) -> None:
        if not self.enabled or self._degraded or self._pool is None:
            return
        with self._lock:
            # Coalesce: last.ckpt is rewritten constantly: if its previous
            # upload is still in flight, skip this one. The next save re-queues
            # it, and teardown always does a final authoritative upload.
            prev = self._futures.get(filepath)
            if prev is not None and not prev.done():
                log.debug("[gdrive] upload of %s still in flight, skipping duplicate", Path(filepath).name)
                return
            self._futures[filepath] = self._pool.submit(self._upload_and_maybe_delete, filepath, may_delete)

    # ------------------------------------------------------- Lightning hooks

    def _save_checkpoint(self, trainer, filepath: str) -> None:
        # 1. Local write first -- parent handles atomic save + loggers + all the
        #    _last_global_step_saved bookkeeping resume depends on.
        super()._save_checkpoint(trainer, filepath)
        if not self.enabled:
            return
        # 2. last.ckpt is uploaded but NOT deleted until teardown (preemption).
        may_delete = not self._is_last_ckpt(filepath)
        if may_delete and self.keep_best_local and filepath == self.best_model_path:
            may_delete = False
        self._submit(filepath, may_delete)

    def _remove_checkpoint(self, trainer, filepath: str) -> None:
        """Rotate a checkpoint out of top-k: drop it locally AND on Drive.

        The local file is usually already gone (we deleted it after upload), so
        only call the strategy when it still exists -- otherwise Lightning logs
        a spurious missing-file warning on every rotation.
        """
        if os.path.exists(filepath):
            super()._remove_checkpoint(trainer, filepath)
        if not self.enabled or self._degraded or self._pool is None:
            return
        remote = self._remote_path(filepath)

        def _purge() -> None:
            try:
                subprocess.run(
                    self._rclone_cmd("deletefile", remote), capture_output=True, text=True, timeout=120
                )
            except Exception as e:
                log.debug("[gdrive] remote purge of %s failed (harmless): %s", remote, e)

        self._pool.submit(_purge)

    def teardown(self, trainer, pl_module, stage: str) -> None:
        """Drain the pool, then do the final last.ckpt upload + cleanup.

        This is the only place last.ckpt's local copy may be removed: by now fit
        has ended, so nothing is going to resume from it.
        """
        try:
            if self.enabled and self._pool is not None:
                with self._lock:
                    pending = [f for f in self._futures.values() if not f.done()]
                for f in pending:
                    try:
                        f.result(timeout=self.upload_timeout_s)
                    except Exception:
                        pass

                last = self._last_checkpoint_saved
                if last and self._is_last_ckpt(last) and os.path.exists(last) and not self._degraded:
                    log.info("[gdrive] final upload of %s before teardown", Path(last).name)
                    self._upload_and_maybe_delete(last, may_delete=True)

                self._pool.shutdown(wait=True)
                self._pool = None
        finally:
            super().teardown(trainer, pl_module, stage)


def resolve_gdrive_cfg(cfg) -> dict | None:
    """Pick this user's Drive block out of ``cfg.checkpoint.gdrive`` using $SDS_WHOAMI.

    Keeps one config working across collaborators with no per-machine edits:
    each of sami/ian/leyang gets their own remote + folder, selected by the
    ``SDS_WHOAMI`` env var exported from the shell rc.
    """
    gd = getattr(cfg.checkpoint, "gdrive", None)
    if gd is None or not gd.get("enabled", False):
        return None
    who = os.environ.get("SDS_WHOAMI")
    if not who:
        log.warning("[gdrive] checkpoint.gdrive.enabled=true but $SDS_WHOAMI is unset -- Drive offload OFF")
        return None
    users = gd.get("users", {}) or {}
    if who not in users:
        log.warning("[gdrive] no checkpoint.gdrive.users entry for SDS_WHOAMI=%r -- Drive offload OFF", who)
        return None
    block = dict(users[who])
    unset = [k for k, v in block.items() if v in ("...", None, "")]
    if unset:
        log.warning("[gdrive] user %r has unconfigured Drive fields %s -- Drive offload OFF", who, unset)
        return None
    merged = {k: v for k, v in gd.items() if k not in ("enabled", "users")}
    merged.update(block)
    binary = merged.get("rclone_binary", "rclone")
    if not (os.path.isabs(binary) and os.path.exists(binary)) and shutil.which(binary) is None:
        log.warning("[gdrive] rclone not found (%r) -- Drive offload OFF", binary)
        return None
    return merged
