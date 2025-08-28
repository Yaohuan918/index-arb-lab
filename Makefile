.PHONY: setup run report test lint

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt
run:
	. .venv/bin/activate && python -m src.index_arb_lab.monitoring.run_once --config configs/strategy.yaml
report:
	. .venv/bin/activate && python -m src.index_arb_lab.monitoring.make_report --date today --out reports/daily_report.html
test:
	. .venv/bin/activate && pytest -q 
replicate:
	. .venv/bin/activate && python -m src.index_arb_lab.replication.run_once --config configs/replication.yaml
replicate-wf:
	. .venv/bin/activate && python -m src.index_arb_lab.replication.walk_forward --config configs/replication.yaml
exec-sim:
	. .venv/bin/activate && python -m src.index_arb_lab.execution.run_once --config configs/execution.yaml
backtest:
	. .venv/bin/activate && python -m src.index_arb_lab.backtest.run_pair --config configs/backtest.yaml