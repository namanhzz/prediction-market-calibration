.PHONY: reproduce kalshi bayesian bayesian-decomp cross-platform robustness revision-diagnostics figures test clean lint format

reproduce: kalshi revision-diagnostics bayesian-decomp bayesian cross-platform robustness figures

kalshi:
	python scripts/run_kalshi.py

# Section 6 measurement-error decomposition (uses the event-clustered cell
# SEs produced by revision-diagnostics).
bayesian-decomp:
	python scripts/run_bayesian_decomp.py

# Supplementary trade-level category random-effects models (M0/M1/M2).
bayesian:
	python scripts/run_bayesian.py

cross-platform:
	python scripts/run_cross_platform.py

robustness:
	python scripts/run_robustness.py

revision-diagnostics:
	python scripts/run_revision_diagnostics.py

figures:
	python scripts/generate_figures.py

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ scripts/ tests/

format:
	ruff format src/ scripts/ tests/

clean:
	rm -rf output/
