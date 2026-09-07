.PHONY: setup models venv contracts anvil bench test demo demo-web-reverse demo-offline corpus-db corpus-footprint verify clean

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

# End-to-end demo. Takes the probe from the webcam unless PROBE= is set:
#   make demo
#   make demo QUERY="official portrait headshot" DEMO_ARGS=--no-pause
#   make demo PROBE=photo.jpg
PROBE ?=
QUERY ?= official portrait headshot high resolution
DEMO_ARGS ?=

demo:
	./scripts/demo.sh $(if $(PROBE),--probe "$(PROBE)") --query "$(QUERY)" $(DEMO_ARGS)

# The same acts, driven by genuine reverse-image search instead. Needs a Cloud
# Vision credential (GOOGLE_VISION_API_KEY or GOOGLE_APPLICATION_CREDENTIALS);
# there is no offline fallback, by design. For an offline dry run:
#   make demo-web-reverse PROVIDER=mock \
#     FACEANCHOR_MOCK_REVERSE_RESPONSE=tests/fixtures/google_web_detection.json
PROVIDER ?= google

demo-web-reverse:
	./scripts/demo.sh $(if $(PROBE),--probe "$(PROBE)") --web-reverse --provider "$(PROVIDER)" $(DEMO_ARGS)

# Offline demo over a locally built corpus — no network at all. Build the
# corpus once (make corpus-db), then this replays it byte-for-byte, which is
# what you want under a restricted network or when the run has to be
# repeatable. `make corpus-footprint` shows every score, not just the winner.
CORPUS_DB ?= evidence/corpus-db
CORPUS ?= mastodon:tag/selfie
PAGES ?= 3

corpus-db:
	$(VENV)/python -m faceanchor.cli corpus-build --db "$(CORPUS_DB)" \
		$(foreach c,$(CORPUS),--corpus "$(c)") --pages $(PAGES)

corpus-footprint:
	$(VENV)/python -m faceanchor.cli corpus-footprint --probe "$(PROBE)" --db "$(CORPUS_DB)" \
		--subject-consent consent/self.example.json --report evidence/footprint.json

demo-offline:
	./scripts/demo.sh $(if $(PROBE),--probe "$(PROBE)") --corpus-db "$(CORPUS_DB)" $(DEMO_ARGS)

verify:
	$(VENV)/python -m faceanchor.cli verify --help

clean:
	rm -rf .venv contracts/out contracts/cache
