#!/usr/bin/env bash
set -euo pipefail

# project root
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# env
source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT/src"

# folders + logging
mkdir -p "$ROOT/logs" "$ROOT/reports"
DATE="$(date +%F)"
LOG="$ROOT/logs/daily_${DATE}.log"   # <— note the $

# log to console + file
exec > >(tee -a "$LOG") 2>&1

echo "[INFO] Starting daily @ $(date) | ROOT=$ROOT | LOG=$LOG"

# (1) Basis monitor
python -m src.index_arb_lab.monitoring.run_once --config "$ROOT/configs/strategy.yaml"

# (2) Replication
python -m src.index_arb_lab.replication.walk_forward --config "$ROOT/configs/replication.yaml"

# (3) Pair backtest
python -m src.index_arb_lab.backtest.run_pair --config "$ROOT/configs/backtest.yaml"

# (4) HTML report (needs --date and --out)
python -m src.index_arb_lab.monitoring.make_report --date today --out "$ROOT/reports/daily_report.html"

# (5) Notify (Slack optional; prints if no webhook)
python "$ROOT/scripts/notify.py" \
  --title "Index-Arb Daily ${DATE}" \
  --text "Run finished. See reports/ and $(basename "$LOG")" \
  --attach "$ROOT/reports/daily_report.html" || true

echo "[INFO] Done @ $(date)"
