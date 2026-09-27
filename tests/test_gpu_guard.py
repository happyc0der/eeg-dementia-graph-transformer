from eegdementia.gpu_guard import parse_model_status, parse_nvidia_smi


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
