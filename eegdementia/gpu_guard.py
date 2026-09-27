"""Yield the GPU to the machine owner's own jobs.

Phase-2 GPU runs happen on a personal laptop whose owner's GPU work always wins. Before
every outer fold (and periodically during long trainings) ``GpuGuard.wait_until_free()``
checks for *foreign* GPU users and, if any is found, frees this process's VRAM (the caller
moves its model to the CPU) and polls every ``poll_s`` seconds until the GPU is free again.
Every pause is logged (and appended to ``pause_log``) so it can be reported.

A foreign user is any of:

* an Ollama server with a model loaded, a standalone llama.cpp server, a Python process
  with CUDA loaded that is not this process (or its parent / children), a WSL job, a game
  or emulator (as reported by the machine's ``model-status`` script, when present);
* GPU utilisation >= 15 % while this process is idle (the check runs synchronously, so our
  own kernels are not running at that moment);
* an ``nvidia-smi`` compute ("C") process that is not ours;
* a Docker container that was not already running when the guard was created.

``model-status`` is the owner's status script: ``$MODEL_STATUS_CMD`` if set, else
``model-status`` on ``PATH``. Without it only the ``nvidia-smi`` checks are used. If
``model-status`` fails or times out, the failure itself is a reason to wait: the owner's
jobs win when we cannot tell.

False positives: a reason that is *only* GPU utilisation is re-checked 30 s later before
pausing, because the first utilisation sample after one of our own folds can still contain
our own load (this caused one self-inflicted pause in phase 2).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil

log = logging.getLogger(__name__)

FAILED = "<failed"
UTIL_REASON = "GPU utilisation"


def find_model_status() -> Path | None:
    """``$MODEL_STATUS_CMD`` (a path, or a command on ``PATH``) if set, else ``model-status``
    found on ``PATH`` (or None)."""
    env = os.environ.get("MODEL_STATUS_CMD")
    if env:
        if Path(env).exists():
            return Path(env)
        found = shutil.which(env)
        if found:
            return Path(found)
        log.warning("MODEL_STATUS_CMD=%r is neither a file nor a command on PATH; not using model-status", env)
        return None
    found = shutil.which("model-status")
    return Path(found) if found else None


def _own_pids() -> set[int]:
    pids = {os.getpid()}
    try:
        p = psutil.Process()
        pids |= {q.pid for q in p.parents()}
        pids |= {q.pid for q in p.children(recursive=True)}
    except (OSError, psutil.Error) as e:  # e.g. a child exited while being listed
        log.warning("could not list parent/child processes: %s", e)
    return pids


def _run(cmd: list[str], timeout: float = 60.0) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace", check=False)
        return r.stdout + "\n" + r.stderr
    except (OSError, subprocess.SubprocessError) as e:  # returned as text; the caller reports it
        return f"{FAILED}: {e}>"


def parse_model_status(text: str, own: set[int], docker_baseline: set[str]) -> list[str]:
    """Reasons (empty = free) from the text output of ``model-status``."""
    reasons = []
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"GPU\s+(\d+)% busy", s)
        if m and int(m.group(1)) >= 15:
            reasons.append(f"{UTIL_REASON} {m.group(1)}% while we are idle")
        if s.startswith("Ollama") and "no model loaded" not in s:
            reasons.append(f"Ollama model loaded: {s}")
        if s.startswith("llama.cpp"):
            reasons.append(s)
        m = re.match(r"Python\s+pid\s+(\d+)", s)
        if m and int(m.group(1)) not in own:
            reasons.append(f"foreign Python GPU job: {s}")
        if s.startswith("WSL"):
            reasons.append(f"WSL job: {s}")
        if s.startswith("Games"):
            reasons.append(f"game running: {s}")
        if s.startswith("Docker"):
            names = re.findall(r"(?:^Docker\s+\d+ container\(s\):\s*|;\s*)([^\s;(]+) \(", s)
            new = [n for n in names if n not in docker_baseline]
            if new:
                reasons.append(f"new Docker container(s): {new}")
    return reasons


def parse_nvidia_smi(text: str, own: set[int]) -> list[str]:
    """Foreign compute-only ("C") processes in the ``nvidia-smi`` process table."""
    reasons = []
    for line in text.splitlines():
        m = re.match(r"\|\s+\d+\s+\S+\s+\S+\s+(\d+)\s+(C|C\+G|G)\s+(.+?)\s+\S+\s*\|", line)
        if m and m.group(2) == "C" and int(m.group(1)) not in own:
            reasons.append(f"foreign CUDA compute process pid {m.group(1)}: {m.group(3).strip()}")
    return reasons


@dataclass
class GpuGuard:
    poll_s: float = 600.0
    min_interval_s: float = 300.0  # for periodic in-training checks
    confirm_s: float = 30.0  # re-check delay for utilisation-only reasons
    pause_log: Path | None = None
    use_model_status: bool = True
    model_status: Path | None = None
    docker_baseline: set[str] = field(default_factory=set)
    _last_check: float = 0.0
    pauses: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.use_model_status:
            self.model_status = self.model_status or find_model_status()
            if self.model_status is None or not Path(self.model_status).exists():
                self.use_model_status = False
        if not self.docker_baseline and shutil.which("docker"):
            out = _run(["docker", "ps", "--format", "{{.Names}}"], timeout=30)
            if not out.startswith(FAILED):
                self.docker_baseline = {x.strip() for x in out.splitlines() if x.strip() and "error" not in x.lower()}
        log.info("GPU guard active (model-status: %s, docker baseline: %s)",
                 self.model_status if self.use_model_status else None, sorted(self.docker_baseline))

    def reasons(self) -> list[str]:
        own = _own_pids()
        reasons = []
        if self.use_model_status:
            cmd = [str(self.model_status)]
            if Path(self.model_status).suffix.lower() in (".cmd", ".bat"):
                cmd = ["cmd", "/c", *cmd]
            out = _run(cmd, timeout=120)
            if out.startswith(FAILED):
                reasons.append(f"model-status failed, cannot tell whether the GPU is free: {out}")
            else:
                reasons += parse_model_status(out, own, self.docker_baseline)
        if shutil.which("nvidia-smi"):
            out = _run(["nvidia-smi"])
            if out.startswith(FAILED):
                log.warning("nvidia-smi failed: %s", out)
            reasons += parse_nvidia_smi(out, own)
        self._last_check = time.time()
        return reasons

    def _confirmed_reasons(self) -> list[str]:
        """``reasons()``, but a pure GPU-utilisation reason must persist ``confirm_s`` later: the
        first utilisation sample right after our own kernels finish still contains our own load."""
        time.sleep(3)
        r = self.reasons()
        if r and all(x.startswith(UTIL_REASON) for x in r):
            time.sleep(self.confirm_s)
            r = self.reasons()
        return r

    def wait_until_free(self, on_pause=None, on_resume=None, context: str = "") -> float:
        """Block while a foreign GPU user exists. Returns the seconds spent paused."""
        r = self._confirmed_reasons()
        if not r:
            return 0.0
        t0 = time.time()
        log.warning("GPU busy (%s) -> pausing: %s", context, "; ".join(r))
        if on_pause is not None:
            on_pause()
        first = r
        while r:
            time.sleep(self.poll_s)
            r = self.reasons()
            if r:
                log.info("still busy: %s", "; ".join(r))
        dt = time.time() - t0
        rec = {"start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), "minutes": round(dt / 60, 1),
               "context": context, "reasons": first}
        self.pauses.append(rec)
        if self.pause_log is not None:
            Path(self.pause_log).parent.mkdir(parents=True, exist_ok=True)
            with open(self.pause_log, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(rec) + "\n")
        log.warning("GPU free again after %.1f min; resuming", dt / 60)
        if on_resume is not None:
            on_resume()
        return dt

    def maybe_wait(self, on_pause=None, on_resume=None, context: str = "") -> float:
        """Rate-limited check for use inside training loops."""
        if time.time() - self._last_check < self.min_interval_s:
            return 0.0
        return self.wait_until_free(on_pause, on_resume, context)
