"""Tampilkan ringkasan metrics.json untuk analisis cepat."""

import json
from pathlib import Path

METRICS_PATH = Path("models/metrics.json")

if not METRICS_PATH.exists():
    raise FileNotFoundError(f"Metrics tidak ditemukan: {METRICS_PATH}")

m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

print("=" * 60)
print("RINGKASAN MODEL")
print("=" * 60)
for k, v in m.items():
    if k == "all_models":
        continue
    if isinstance(v, float) and v > 1e6:
        print(f"{k:25s}: Rp {v:,.0f}")
    else:
        print(f"{k:25s}: {v}")

print()
print("=" * 60)
print("PERBANDINGAN MODEL")
print("=" * 60)
for name, info in m["all_models"].items():
    cv_mape = info["cv_mape"]
    print(f"{name:20s}: MAPE CV (log-space) = {cv_mape:.4f}")
    print(f"{'':20s}  Best params: {info['best_params']}")