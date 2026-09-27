from eegdementia import gpu_guard as gg
from eegdementia.gpu_guard import GpuGuard, parse_model_status, parse_nvidia_smi


def test_model_status_parsing():
    idle = """
  GPU        0% busy | VRAM 1.4/16 GB | 31 W | 32 C
  Ollama   :11434  no model loaded
  Docker   1 container(s): quirky_bell (swebench-sandbox:latest, Up 24 hours)

  IDLE - nothing is running on the GPU. Safe to restart.
"""
    assert parse_model_status(idle, own={1}, docker_baseline={"quirky_bell"}) == []
    busy = """
  GPU       40% busy | VRAM 9.4/16 GB | 131 W | 62 C
  Ollama   :11434  qwen3:8b  (5.1 GB on GPU, stays loaded until 03:10)
  Python   pid 123 (GPU, running 3m): python train.py
  Python   pid 7 (GPU, running 3m): python run_phase2.py
  Games    rpcs3  (GPU load from games is ignored)
  Docker   2 container(s): quirky_bell (a, Up 1 h); newone (b, Up 1m)
"""
    r = parse_model_status(busy, own={7}, docker_baseline={"quirky_bell"})
    text = " | ".join(r)
    assert "GPU utilisation 40%" in text
    assert "Ollama model loaded" in text
    assert "pid 123" in text and "pid 7" not in text  # our own process is not foreign
    assert "game running" in text
    assert "newone" in text and "quirky_bell" not in text.split("new Docker")[1]


def test_nvidia_smi_parsing():
    table = (
        "|    0   N/A  N/A            2284    C+G   C:\\Windows\\System32\\dwm.exe           N/A      |\n"
        "|    0   N/A  N/A             999      C   C:\\x\\python.exe                     1200MiB |\n"
        "|    0   N/A  N/A              55      C   C:\\ours\\python.exe                  800MiB |\n"
    )
    r = parse_nvidia_smi(table, own={55})
    assert len(r) == 1 and "999" in r[0]


class _ScriptedGuard(GpuGuard):
    """GpuGuard whose ``reasons()`` returns a scripted sequence (no subprocesses)."""

    def __init__(self, script, **kw):
        self.script = list(script)
        self.calls = 0
        super().__init__(use_model_status=False, docker_baseline={"x"}, poll_s=0, confirm_s=0, **kw)

    def reasons(self):
        self.calls += 1
        return self.script.pop(0) if self.script else []


def test_utilisation_false_positive_is_rechecked_not_paused(monkeypatch):
    monkeypatch.setattr(gg.time, "sleep", lambda s: None)
    # our own fold's load still shows in the first sample, gone 30 s later -> no pause
    g = _ScriptedGuard([["GPU utilisation 17% while we are idle"], []])
    paused = []
    assert g.wait_until_free(on_pause=lambda: paused.append(1)) == 0.0
    assert g.calls == 2 and not paused and not g.pauses


def test_persistent_utilisation_and_foreign_jobs_pause(monkeypatch):
    monkeypatch.setattr(gg.time, "sleep", lambda s: None)
    events = []
    g = _ScriptedGuard([["GPU utilisation 60% while we are idle"], ["GPU utilisation 60% while we are idle"], []])
    g.wait_until_free(on_pause=lambda: events.append("pause"), on_resume=lambda: events.append("resume"))
    assert events == ["pause", "resume"] and len(g.pauses) == 1
    # a non-utilisation reason pauses at once, without the confirmation re-check
    g = _ScriptedGuard([["Ollama model loaded: x", "GPU utilisation 60% while we are idle"], []])
    g.wait_until_free()
    assert g.calls == 2 and len(g.pauses) == 1


def test_failed_model_status_is_a_reason(monkeypatch, tmp_path):
    script = tmp_path / "model-status.cmd"
    script.write_text("@echo off\n")
    monkeypatch.setattr(gg, "_run", lambda cmd, timeout=60.0: f"{gg.FAILED}: timed out>")
    monkeypatch.setattr(gg.shutil, "which", lambda name: None)  # no docker, no nvidia-smi
    g = GpuGuard(model_status=script)
    assert g.use_model_status
    r = g.reasons()
    assert len(r) == 1 and "model-status failed" in r[0]


def test_model_status_lookup(monkeypatch, tmp_path):
    script = tmp_path / "ms.cmd"
    script.write_text("@echo off\n")
    monkeypatch.setenv("MODEL_STATUS_CMD", str(script))
    assert gg.find_model_status() == script
    # a command name is looked up on PATH; a name that resolves to nothing disables model-status
    monkeypatch.setattr(gg.shutil, "which", lambda name: str(script) if name == "my-status" else None)
    monkeypatch.setenv("MODEL_STATUS_CMD", "my-status")
    assert gg.find_model_status() == script
    monkeypatch.setenv("MODEL_STATUS_CMD", str(tmp_path / "missing.cmd"))
    assert gg.find_model_status() is None
    monkeypatch.delenv("MODEL_STATUS_CMD")
    assert gg.find_model_status() is None
    assert GpuGuard(docker_baseline={"x"}).use_model_status is False


def test_own_pids_survives_psutil_errors(monkeypatch):
    class Racy:
        def parents(self):
            return []

        def children(self, recursive=False):
            raise gg.psutil.NoSuchProcess(12345)  # a child exited while being listed

    monkeypatch.setattr(gg.psutil, "Process", lambda: Racy())
    assert gg.os.getpid() in gg._own_pids()
