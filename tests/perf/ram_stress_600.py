"""600-turn chat simulation: realistic per-turn sizes (tool results ~2-8KB,
model replies ~1-2KB), plus periodic session saves. Measures RSS peak.
Reuses the ram_stress harness env/stub-server setup.
"""
import contextlib
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.update(
    NX_ENV_FILE="/dev/null",
    NX_SKIP_STARTUP_PROBES="1",
    NX_LOG_REQUESTS="0",
    NX_APPROVAL="yolo",
    # defaults: spill + autocompress ON, as in a real chat
)

import nx as _m
from nx import _state as st
from nx._batch import run_batch
from nx._eval_cmd import SCENARIOS, _stub_server

OUT = Path(__file__).parent / ".nx-ram-600"
OUT.mkdir(exist_ok=True)


def current_rss_kb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * (os.sysconf("SC_PAGE_SIZE") // 1024)


samples, stop = [], threading.Event()


def sampler():
    while not stop.is_set():
        samples.append(current_rss_kb())
        time.sleep(0.25)


N = 600
script = []
for i in range(N):
    # alternate: a medium read (the read_file tool result lands in transcript)
    # and a bare model turn (assistant text only)
    if i % 2 == 0:
        script.append({"tool_call": {"name": "read_file", "args": {
            "path": "med.txt", "offset": (i * 37) % 4000 + 1, "limit": 80}}})
    else:
        script.append({"tool_call": {"name": "write_file", "args": {
            "path": f"out/n{i}.txt", "content": "y" * 1500}}})
script.append({"final": "done"})

instr = "RAM600: alternate medium reads and small writes until told to stop"
SCENARIOS["ram-600"] = {"title": "600 turns", "instruction": instr, "script": script}

srv, base_url = _stub_server()
threading.Thread(target=srv.serve_forever, daemon=True).start()

(OUT / "workspace").mkdir(exist_ok=True)
(OUT / "workspace" / "med.txt").write_text(
    "".join(f"line {i}: {i*7919%99991} lorem ipsum dolor sit {i}\n" for i in range(5000)))

saved = (st.ACTIVE, st.client, st.MODEL, dict(st.SOURCES), list(st.SOURCE_ORDER))
old_cwd = os.getcwd()
try:
    src = _m._sources.Source("ram-stub", base_url, "sk-h", "ram-stub")
    src.client = _m._sources.OpenAI(base_url=base_url, api_key="sk-h")
    src._context_window = None
    src._context_window_probe_attempted = True
    st.SOURCES["ram-stub"] = src
    st.SOURCE_ORDER.append("ram-stub")
    st.ACTIVE, st.client, st.MODEL, st.YOLO = src, src.client, "ram-stub", True

    tasks = OUT / "tasks.jsonl"
    tasks.write_text(json.dumps({"id": "t600", "instruction": instr,
                                 "timeout_seconds": 0}) + "\n")
    os.chdir(OUT / "workspace")
    t0 = time.time()
    threading.Thread(target=sampler, daemon=True).start()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = run_batch(["-f", str(tasks), "--max-turns", "99999",
                          "--parallel", "1", "--continue-on-error",
                          "--yolo", "--json"])
    elapsed = time.time() - t0
    stop.set()
    time.sleep(0.3)
    for line in buf.getvalue().splitlines():
        try:
            r = json.loads(line)
            print(f"task: status={r.get('status')} turns={r.get('turns')} "
                  f"elapsed={r.get('elapsed_seconds')}s err={str(r.get('error'))[:150]}")
        except json.JSONDecodeError:
            print(line)
    print(f"exit={code} wall={elapsed:.1f}s  RSS start={samples[0]/1024:.0f} MB "
          f"peak={max(samples)/1024:.0f} MB end={samples[-1]/1024:.0f} MB")
finally:
    stop.set()
    os.chdir(old_cwd)
    srv.shutdown()
    srv.server_close()
    st.SOURCES.clear()
    st.SOURCES.update(saved[3])
    st.SOURCE_ORDER[:] = saved[4]
    st.ACTIVE, st.client, st.MODEL = saved[0], saved[1], saved[2]
