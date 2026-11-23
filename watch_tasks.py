import sys
import time
import datetime as dt

import ee
import config

ee.Initialize(project=config.PROJECT)

ACTIVE = {"RUNNING", "PENDING", "READY", "SUBMITTED"}
MARK = {"SUCCEEDED": "✓", "COMPLETED": "✓", "FAILED": "✗",
        "CANCELLED": "-", "RUNNING": "▸", "PENDING": "·", "READY": "·"}


# How many tasks to PRINT. The counts and the EECU total below are computed
# over every operation, not this window — truncating before tallying is what
# made this tool under-report August 2026 usage as 196 EECU-hours when the
# account had actually burned 234.2 across 133 operations. A quota tool that
# silently reads the first 40 rows is worse than no quota tool.
SHOW = 40


def snapshot():
    ops = ee.data.listOperations() or []
    rows, active, done, failed = [], 0, 0, 0
    total_eecu, n_ops = 0.0, len(ops)

    for o in ops:
        mm = o.get("metadata", {})
        try:
            total_eecu += float(mm.get("batchEecuUsageSeconds") or 0)
        except (TypeError, ValueError):
            pass

    for o in ops[:SHOW]:
        m = o.get("metadata", {})
        state = m.get("state", "?")
        desc = m.get("description", "?")

        start, end = m.get("startTime"), m.get("endTime")
        elapsed = "—"
        if start:
            s = dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
            e = (dt.datetime.fromisoformat(end.replace("Z", "+00:00"))
                 if end else dt.datetime.now(dt.timezone.utc))
            elapsed = f"{(e - s).total_seconds() / 60:.1f}m"

        eecu = m.get("batchEecuUsageSeconds")
        err = (m.get("error", {}) or {}).get("message", "")

        if state in ACTIVE:
            active += 1
        elif state in ("SUCCEEDED", "COMPLETED"):
            done += 1
        elif state == "FAILED":
            failed += 1

        rows.append((MARK.get(state, "?"), desc[:24], state, elapsed,
                     f"{eecu:.0f}" if eecu else "—", err[:44]))

    return rows, active, done, failed, total_eecu, n_ops


def render():
    rows, active, done, failed, total_eecu, n_ops = snapshot()
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    print(f"\n  {stamp}   {done} done · {active} active · {failed} failed"
          f"   (of {len(rows)} shown / {n_ops} total)\n")
    # The noncommercial Community tier is 150 EECU-hours per calendar month,
    # resetting on the 1st. Past it you are not cut off, only throttled.
    hrs = total_eecu / 3600
    bar = "over quota — restricted mode" if hrs > 150 else f"{150 - hrs:.0f} h left this month"
    print(f"  lifetime EECU {total_eecu:,.0f} s = {hrs:.1f} h   ·   {bar}\n")
    print(f"  {'':2} {'task':<24} {'state':<11} {'time':>6} {'EECU-s':>8}")
    print("  " + "-" * 62)
    for mark, desc, state, elapsed, eecu, err in rows:
        print(f"  {mark:2} {desc:<24} {state:<11} {elapsed:>6} {eecu:>8}")
        if err:
            print(f"     └─ {err}")
    return active


if "-w" in sys.argv:
    try:
        while True:
            if render() == 0:
                print("\n  nothing active — all tasks finished\n")
                break
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n  stopped watching (tasks keep running on Google's side)\n")
else:
    render()
    print()
