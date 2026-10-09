# One command from the committed results to the paper's numbers, figures and PDF.
#   make numbers   paper/generated/{numbers.tex,numbers.json,tab_*.tex}
#   make figures   paper/figures/*.pdf
#   make paper     both, then paper/build/main.pdf
#   make test      the test suite
#   make smoke     the smoke cell on the CPU: every stage, the tiny model trained for real
#   make check     doc paths, secrets scan, and the registry (every record loads and checks)
#   make rebuttal  the submitted cell's analysis/*.json (discrimination, leakage, questions)
#   make cells     results/cross_cell/*.json: added-cell values, paired tests, tau and recipe
#                  (needs the ignored host copies in the 2026-07 cells)
#   make kappa     $(MAIN_CELL)/validation/kappa_results.json (needs RATER_FILES)
#   make derived   the tracked score copies, from the large ignored files they copy

PYTHON ?= python3
# Inputs kept out of git; point these at the files if they live elsewhere.
# HealthBench-Hard is public; python -m bodhi_lora fetch healthbench downloads it. numbers and figures stop
# without it, because the paper uses its theme tags and rubric points.
HB_HARD ?= data/raw/healthbench/hard.jsonl
# The submitted cell (results/PROVENANCE.md).
MAIN_CELL ?= results/healthbench__medgemma-27b-text__bodhi__2026-05
# The seed-7 training traces (49 MB, ignored). make numbers reads only their scores, from
# the tracked $(MAIN_CELL)/seed_7/filter/filter_scores.json; kappa and derived need the file.
FILTER_TRAIN ?= $(MAIN_CELL)/seed_7/filter/train.jsonl
POSTED_VALUES ?= rebuttal/posted_values.json

.PHONY: numbers figures paper test smoke check rebuttal cells kappa derived

numbers: $(HB_HARD)
	$(PYTHON) scripts/paper_stats.py --healthbench "$(HB_HARD)" --posted-values "$(POSTED_VALUES)"

figures: $(HB_HARD)
	$(PYTHON) scripts/build_paper_figures.py --hb-hard "$(HB_HARD)"

# Runs only when the file is missing: stop with the fix instead of a traceback.
$(HB_HARD):
	@echo "$(HB_HARD) not found. Run python -m bodhi_lora fetch healthbench, or pass HB_HARD=<path>."
	@exit 1

paper: numbers figures
	$(PYTHON) scripts/check_paper_macros.py
	mkdir -p paper/build
	tectonic -X compile --outdir paper/build paper/main.tex

test:
	$(PYTHON) -m pytest -q -p no:cacheprovider

# Writes under SMOKE_ROOT, never under results/, so a smoke run leaves no trace in git.
SMOKE_CELL ?= tiny-rubric__tiny-random-llama__bodhi
SMOKE_ROOT ?= outputs/smoke
smoke:
	$(PYTHON) -m bodhi_lora run $(SMOKE_CELL) --stub-engine --results-root $(SMOKE_ROOT) --force

# list loads every record and runs the family rule before it prints, so a broken
# record fails here with all its problems listed.
check:
	bash scripts/check_no_secrets.sh
	$(PYTHON) scripts/check_doc_paths.py
	$(PYTHON) -m bodhi_lora list cells > /dev/null

rebuttal: $(HB_HARD)
	$(PYTHON) rebuttal/analyses/mimicry_split.py --healthbench "$(HB_HARD)"
	$(PYTHON) rebuttal/analyses/interference_mechanism.py
	$(PYTHON) rebuttal/analyses/second_person_questions.py

# Reads the added cells' raw eval JSON and host logs in their results/ folders, copied
# from the GPU host and gitignored (REPRODUCIBILITY.md section 7). cell_values.py diffs
# the values against $(POSTED_VALUES); cell_recipe.py reads tau, trace counts and the
# adapter recipe per cell (PROV-3).
cells:
	$(PYTHON) rebuttal/analyses/cell_values.py --posted "$(POSTED_VALUES)"
	$(PYTHON) rebuttal/analyses/quality_drop.py
	$(PYTHON) rebuttal/analyses/cell_recipe.py

# The returned rater files carry names, so they stay outside the repo; pass their
# paths, e.g. make kappa RATER_FILES="/path/to/Grading-<rater>.xlsx".
kappa:
	@test -n '$(RATER_FILES)' || { echo "set RATER_FILES to the returned Grading-*.xlsx path(s)"; exit 1; }
	$(PYTHON) scripts/compute_irr_kappa.py --rater-files $(RATER_FILES) --filter-traces "$(FILTER_TRAIN)"

derived:
	$(PYTHON) scripts/export_derived_inputs.py --filter-train "$(FILTER_TRAIN)"
