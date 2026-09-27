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

On machines without ``model-status`` only the ``nvidia-smi`` checks are used.
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

log = logging.getLogger(__name__)

MODEL_STATUS = Path(os.environ.get("MODEL_STATUS_CMD", r"C:\AI\bin\model-status.cmd"))


def _own_pids() -> set[int]:
    pids = {os.getpid()}
    try:
        import psutil

        p = psutil.Process()
        pids |= {q.pid for q in p.parents()}
        pids |= {q.pid for q in p.children(recursive=True)}
    except Exception:  # pragma: no cover
        pass
    return pids


def _run(cmd: list[str], timeout: float = 60.0) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace")
        return r.stdout + "\n" + r.stderr
    except Exception as e:  # pragma: no cover - reported as a reason, never swallowed silently
        return f"<failed: {e}>"


def parse_model_status(text: str, own: set[int], docker_baseline: set[str]) -> list[str]:
    """Reasons (empty = free) from the text output of ``model-status``."""
    reasons = []
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"GPU\s+(\d+)% busy", s)
        if m and int(m.group(1)) >= 15:
            reasons.append(f"GPU utilisation {m.group(1)}% while we are idle")
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
    pause_log: Path | None = None
    use_model_status: bool = True
    docker_baseline: set[str] = field(default_factory=set)
    _last_check: float = 0.0
    pauses: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.use_model_status and not MODEL_STATUS.exists():
            self.use_model_status = False
        if not self.docker_baseline and shutil.which("docker"):
            out = _run(["docker", "ps", "--format", "{{.Names}}"], timeout=30)
            if not out.startswith("<failed"):
                self.docker_baseline = {x.strip() for x in out.splitlines() if x.strip() and "error" not in x.lower()}
        log.info("GPU guard active (model-status: %s, docker baseline: %s)", self.use_model_status, sorted(self.docker_baseline))

    def reasons(self) -> list[str]:
        own = _own_pids()
        reasons = []
        if self.use_model_status:
            reasons += parse_model_status(_run(["cmd", "/c", str(MODEL_STATUS)], timeout=120), own, self.docker_baseline)
        if shutil.which("nvidia-smi"):
            reasons += parse_nvidia_smi(_run(["nvidia-smi"]), own)
        self._last_check = time.time()
        return reasons

    def wait_until_free(self, on_pause=None, on_resume=None, context: str = "") -> float:
        """Block while a foreign GPU user exists. Returns the seconds spent paused."""
        r = self.reasons()
        if not r:
            return 0.0
        t0 = time.time()
        log.warning("GPU busy (%s) -> pausing %s: %s", context, "", "; ".join(r))
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
            with open(self.pause_log, "a") as fh:
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
