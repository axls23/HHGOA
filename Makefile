.PHONY: setup models venv contracts anvil bench test demo verify clean

FOUNDRY_BIN := $(HOME)/.foundry/bin
export PATH := $(FOUNDRY_BIN):$(PATH)

VENV := .venv/bin

setup: venv models contracts
	@echo "setup complete"

venv:
	uv venv --python 3.12 .venv
	uv pip install -e ".[dev]" --python $(VENV)/python

models:
	./models/fetch.sh

contracts:
	@command -v forge >/dev/null 2>&1 || curl -fsSL https://foundry.paradigm.xyz | bash
	@command -v forge >/dev/null 2>&1 || (echo "run 'foundryup' then re-run make contracts" && exit 1)
	cd contracts && forge build

anvil:
	anvil

bench:
	taskset -c 0-15 $(VENV)/python scripts/bench.py -n 300

test:
	cd contracts && forge test
	$(VENV)/python -m pytest tests/ -q

demo:
	$(VENV)/python -m faceanchor.cli run --help

verify:
	$(VENV)/python -m faceanchor.cli verify --help

clean:
	rm -rf .venv contracts/out contracts/cache
