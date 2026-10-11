"""Instrumented variant: task A only, tracemalloc snapshot at the end."""
import contextlib
import json
import os
import shutil
import sys
import threading
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(
    NX_ENV_FILE="/dev/null", NX_SKIP_STARTUP_PROBES="1", NX_LOG_REQUESTS="0",
    NX_AUTOCOMPRESS_PERCENT="0", NX_APPROVAL="yolo", NX_TOOL_RESULT_SPILL="0",
)

import nx as _m
from nx import _state as st
from nx._batch import run_batch
from nx._eval_cmd import SCENARIOS, _stub_server

OUT = Path(__file__).parent / ".nx-ram-stress2"
WS = OUT / "workspace"


def big_file(path: Path, lines: int) -> None:
    with open(path, "w") as f:
        for i in range(lines):
            f.write(f"line {i}: {i * 7919 % 999983} alpha beta gamma delta {i}\n")


if OUT.exists():
    shutil.rmtree(OUT)
WS.mkdir(parents=True)
big_file(WS / "big.txt", 12000)
big_file(WS / "big2.txt", 12000)

instr = "RAMSTRESS-A2: read the big files repeatedly until told to stop"
script = []
for i in range(30):
    script.append({"tool_call": {"name": "read_file", "args": {
        "path": "big.txt" if i % 2 == 0 else "big2.txt",
        "offset": (i % 9) * 1200 + 1, "limit": 25000}}})
script.append({"final": "done reading"})
SCENARIOS["ram-a2"] = {"title": "big transcript", "instruction": instr,
                       "script": script}

srv, base_url = _stub_server()
threading.Thread(target=srv.serve_forever, daemon=True).start()

saved = (st.ACTIVE, st.client, st.MODEL, dict(st.SOURCES), list(st.SOURCE_ORDER))
old_cwd = os.getcwd()
os.environ["NX_HTTP_TIMEOUT_READ"] = "30"
tracemalloc.start(1)
try:
    src = _m._sources.Source("ram-stub", base_url, "sk-h", "ram-stub")
    src.client = _m._sources.OpenAI(base_url=base_url, api_key="sk-h")
    src._context_window = None
    src._context_window_probe_attempted = True
    st.SOURCES["ram-stub"] = src
    st.SOURCE_ORDER.append("ram-stub")
    st.ACTIVE = src; st.client = src.client; st.MODEL = "ram-stub"; st.YOLO = True

    tasks = OUT / "tasks.jsonl"
    with open(tasks, "w") as f:
        f.write(json.dumps({"id": "big-transcript", "instruction": instr,
                            "timeout_seconds": 0}) + "\n")
    os.chdir(WS)
    results = OUT / "results.jsonl"
    t0 = time.time()
    with open(results, "w") as rf, contextlib.redirect_stdout(rf):
        code = run_batch(["-f", str(tasks), "--max-turns", "99999",
                          "--parallel", "1", "--continue-on-error",
                          "--yolo", "--json"])
    with open("/proc/self/statm") as f:
        rss = int(f.read().split()[1]) * 4 / 1024
    print(f"wall {time.time()-t0:.1f}s  RSS {rss:.0f} MB")

    snap = tracemalloc.take_snapshot()
    print("\ntop tracemalloc sites (current):")
    for s in snap.statistics("lineno")[:15]:
        print(f"  {s.size/1048576:8.1f} MB  {s}")
finally:
    os.chdir(old_cwd)
    srv.shutdown(); srv.server_close()
    st.SOURCES.clear(); st.SOURCES.update(saved[3]); st.SOURCE_ORDER[:] = saved[4]
    st.ACTIVE, st.client, st.MODEL = saved[0], saved[1], saved[2]
