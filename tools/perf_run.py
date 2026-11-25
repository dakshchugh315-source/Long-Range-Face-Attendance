"""Real performance measurement on THIS laptop (no camera needed).

python tools/perf_run.py [seconds]          (default 180)

Runs the real Engine (detection + tiling + recognition + voting + auto-zoom) against the
simulated phone from tests/test_autozoom.py (a 4K classroom with 6 far-away students)
and samples every 10 s: FPS, per-stage time, process memory (working set) and CPU.
Memory that keeps growing = leak. Writes tools/perf_result.json.
"""
import ctypes
import ctypes.wintypes as wt
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_autozoom as T  # noqa: E402  (temp DB, fake phone, classroom builder)


class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def mem_mb():
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.GetCurrentProcess.restype = wt.HANDLE                     # 64-bit handle, not int
    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
    psapi.GetProcessMemoryInfo.restype = wt.BOOL
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return round(pmc.WorkingSetSize / 2 ** 20)


def main():
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    import app as appmod
    appmod.get_app()
    T.Phone.room, placed = T.build_room(None, T.pick_people(6))
    threading.Thread(target=T.Server(("127.0.0.1", T.PHONE), T.Phone).serve_forever, daemon=True).start()
    eng = appmod.engine
    eng.start(f"http://127.0.0.1:{T.PHONE}/video", "perf", autozoom=True)
    samples, t0 = [], time.time()
    cpu0, wall0 = time.process_time(), time.time()
    while time.time() - t0 < seconds and eng.running:
        time.sleep(10)
        cpu1, wall1 = time.process_time(), time.time()
        s = {"t": round(time.time() - t0), "fps": round(eng.fps, 2), "mem_mb": mem_mb(),
             "cpu_pct_of_machine": round(100 * (cpu1 - cpu0) / (wall1 - wall0) / os.cpu_count()),
             "timing_ms": dict(eng.timing), "marked": len(eng.present),
             "camera": eng.autozoom.camera_state() if eng.autozoom else None}
        cpu0, wall0 = cpu1, wall1
        samples.append(s)
        print(s)
    err = eng.error
    eng.stop()
    mems = [s["mem_mb"] for s in samples]
    half = len(mems) // 2
    res = {"seconds": seconds, "error": err, "students_in_room": placed, "samples": samples,
           "fps_avg": round(sum(s["fps"] for s in samples) / max(1, len(samples)), 2),
           "mem_first_half_avg": round(sum(mems[:half]) / max(1, half)),
           "mem_second_half_avg": round(sum(mems[half:]) / max(1, len(mems) - half)),
           "marked": samples[-1]["marked"] if samples else 0}
    res["mem_growth_mb"] = res["mem_second_half_avg"] - res["mem_first_half_avg"]
    out = Path(__file__).with_name("perf_result.json")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nFPS avg {res['fps_avg']} | memory {res['mem_first_half_avg']} -> {res['mem_second_half_avg']} MB "
          f"(growth {res['mem_growth_mb']} MB) | marked {res['marked']}/6 | error {err!r}\nsaved {out}")


if __name__ == "__main__":
    main()
