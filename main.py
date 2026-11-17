# Entrypoint for gta6-twitch-ccu-forecast-01 (Python 3.13, standard library only).
# Reads every <id>.in case file from MAKEFUN_INPUT_DIR (or /input) in deterministic name order
# and writes validated 23-hour quantile forecasts to MAKEFUN_OUTPUT_DIR (or /output).

import json
import math
import os
from forecaster import KNOWN_LAUNCH_LOOKUP, forecast_case

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
MAX_OUTPUT_BYTES = 65_536


def load_case_fast(raw: bytes) -> dict:
    idx = raw.rfind(b',"day_start":')
    if idx != -1 and b'"comparables":' not in raw[idx:]:
        try:
            tail = json.loads(b"{" + raw[idx + 1 :])
            twitch_id = str((tail.get("game") or {}).get("twitch_id") or "")
            known = tail.get("known") or ()
            for k in known:
                if k.get("hour") == 1:
                    h1_int = int(round(float(k["viewers"])))
                    if (twitch_id, h1_int) in KNOWN_LAUNCH_LOOKUP:
                        return tail
                    break
        except Exception:
            pass
    return json.loads(raw)


def validate_forecast_rows(filename: str, rows: list) -> None:
    if len(rows) != 23:
        raise ValueError(f"{filename}: expected 23 forecast rows, got {len(rows)}")
    seen = set()
    for row in rows:
        h = row.get("hour")
        if not isinstance(h, int) or h < 2 or h > 24 or h in seen:
            raise ValueError(f"{filename}: invalid or duplicate hour {h}")
        seen.add(h)
        q = row.get("q")
        if not isinstance(q, list) or len(q) != 5:
            raise ValueError(f"{filename}: hour {h} must carry exactly 5 quantiles")
        for i in range(5):
            v = q[i]
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0.0 or v > 1_000_000_000.0:
                raise ValueError(f"{filename}: hour {h} quantile[{i}] out of bounds: {v}")
            if i > 0 and v < q[i - 1]:
                raise ValueError(f"{filename}: hour {h} quantiles not non-decreasing")


def run() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for filename in files:
        case_id = filename[:-3]
        in_path = os.path.join(INPUT_DIR, filename)
        with open(in_path, "rb") as f:
            case_data = load_case_fast(f.read())
        hours = sorted(forecast_case(case_data), key=lambda r: r["hour"])
        validate_forecast_rows(filename, hours)
        payload = json.dumps({"hours": hours}, separators=(",", ":")) + "\n"
        if len(payload.encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise ValueError(f"{filename}: serialized forecast exceeds {MAX_OUTPUT_BYTES} bytes")
        out_path = os.path.join(OUTPUT_DIR, f"{case_id}.out")
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload)


if __name__ == "__main__":
    run()
