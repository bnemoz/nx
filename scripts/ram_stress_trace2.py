"""RAM stress #3: sampler takes tracemalloc snapshots DURING and AFTER the
batch, logging top allocation sites per sample, to attribute the RSS climb
that continues after the tasks finish."""
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

OUT = Path(__file__).parent / ".nx-ram-stress3"
WS = OUT / "workspace"
T0 = time.time()


def rss_kb() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4


def big_file(path: Path, lines: int) -> None:
    with open(path, "w") as f:
        for i in range(lines):
            f.write(f"line {i}: {i * 7919 % 999983} alpha beta gamma delta {i}\n")


if OUT.exists():
    shutil.rmtree(OUT)
WS.mkdir(parents=True)
big_file(WS / "big.txt", 12000)
big_file(WS / "big2.txt", 12000)

instr = "RAMSTRESS-A3: read the big files repeatedly until told to stop"
script = [{"tool_call": {"name": "read_file", "args": {
    "path": "big.txt" if i % 2 == 0 else "big2.txt",
    "offset": (i % 9) * 1200 + 1, "limit": 25000}}} for i in range(30)]
script.append({"final": "done reading"})
SCENARIOS["ram-a3"] = {"title": "big transcript", "instruction": instr,
                       "script": script}

srv, base_url = _stub_server()
threading.Thread(target=srv.serve_forever, daemon=True).start()

saved = (st.ACTIVE, st.client, st.MODEL, dict(st.SOURCES), list(st.SOURCE_ORDER))
old_cwd = os.getcwd()
os.environ["NX_HTTP_TIMEOUT_READ"] = "30"

stop = threading.Event()
log = open(OUT / "samples.log", "w")
log.write("tracemalloc started\n")


def sampler():
    while not stop.is_set():
        rss = rss_kb()
        snap = tracemalloc.take_snapshot()
        top = snap.statistics("lineno")[:3]
        log.write(f"t+{time.time()-T0:6.1f}s RSS {rss/1024:7.0f} MB\n")
        for s in top:
            log.write(f"    {s.size/1048576:8.1f} MB  {s}\n")
        del snap
        time.sleep(1.0)


tracemalloc.start(1)
sampler_t = threading.Thread(target=sampler, daemon=True)
sampler_t.start()
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
    with open(results, "w") as rf, contextlib.redirect_stdout(rf):
        code = run_batch(["-f", str(tasks), "--max-turns", "99999",
                          "--parallel", "1", "--continue-on-error",
                          "--yolo", "--json"])
    log.write("BATCH DONE\n")
    time.sleep(8)   # watch what climbs after the batch returns
    log.write("IDLE WATCH DONE\n")
finally:
    stop.set()
    sampler_t.join(timeout=3)
    os.chdir(old_cwd)
    srv.shutdown(); srv.server_close()
    st.SOURCES.clear(); st.SOURCES.update(saved[3]); st.SOURCE_ORDER[:] = saved[4]
    st.ACTIVE, st.client, st.MODEL = saved[0], saved[1], saved[2]
    log.close()
print("done; samples:")
print(open(OUT / "samples.log").read())
