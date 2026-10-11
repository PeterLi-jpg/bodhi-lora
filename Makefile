# One command from the committed results to the paper's numbers, figures and PDF.
#   make report    paper/generated/{numbers.tex,numbers.json,tab_*.tex}, from each cell's
#                  analysis/stats.json and the cross-cell statistics (make numbers is the same)
#   make figures   paper/figures/*.pdf, for the first detail cell of configs/report.yaml
#   make paper     both, then paper/build/main.pdf
#   make test      the test suite
#   make smoke     the smoke cell on the CPU: every stage, the tiny model trained for real
#   make check     doc paths, secrets scan, and the registry (every record loads and checks)
#   make extract   each analyzed cell's seed_*/table.csv from its raw outputs, then make analyze
#   make analyze   results/<cell>/analysis/stats.json for every cell with tables, and the
#                  cross-cell statistics once every cell they need has its stats
#   make rebuttal  the submitted cell's analysis/withdrawn/*.json of the three discussion
#                  scripts: the record of how posted values were made; the paper reads none
#   make cells     results/cross_cell/cell_recipe.json from the host logs: the record of the
#                  2026-07 filter counts (needs the ignored host logs)
#   make kappa     $(MAIN_CELL)/validation/kappa_results.json (needs RATER_FILES and ANSWER_KEY)
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

.PHONY: report numbers figures paper test smoke check rebuttal cells kappa derived extract analyze

# The report reads only the tracked analysis outputs, so it needs nothing outside git.
report numbers:
	$(PYTHON) -m bodhi_lora report

# Figure E.1 and the per-seed figure read the cell's stats.json, as the tables do; the
# difficulty, axis and score figures read its tracked response tables (seed_*/table.csv),
# so the figures need nothing outside git but the HealthBench file.
figures: $(HB_HARD)
	$(PYTHON) scripts/build_paper_figures.py --hb-hard "$(HB_HARD)"

# Runs only when the file is missing: stop with the fix instead of a traceback.
$(HB_HARD):
	@echo "$(HB_HARD) not found. Run python -m bodhi_lora fetch healthbench, or pass HB_HARD=<path>."
	@exit 1

paper: report figures
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

# The uniform analysis (docs/ARCHITECTURE.md section 7). extract reads the raw stage
# outputs, which git ignores except for the 2026-05 cell; analyze reads only the tracked
# tables, so it runs in CI, and make report reads only what analyze writes.
extract:
	$(PYTHON) -m bodhi_lora analyze --all --extract

analyze:
	$(PYTHON) -m bodhi_lora analyze --all

# The three scripts are kept as the record of how the posted values were made. Each
# decides meaning by keywords somewhere (docs/CONVENTIONS.md rule 15), so the paper reads
# none of their outputs (rebuttal/withdrawals.md W23 to W25; tests/test_report.py). They
# write into the cell's analysis/withdrawn/.
rebuttal: $(HB_HARD)
	@echo "make rebuttal remakes a record of withdrawn keyword measures (rebuttal/withdrawals.md W23 to W25); the paper reads none of these files."
	$(PYTHON) rebuttal/analyses/mimicry_split.py --healthbench "$(HB_HARD)"
	$(PYTHON) rebuttal/analyses/interference_mechanism.py
	$(PYTHON) rebuttal/analyses/second_person_questions.py

# Reads the host logs of the added cells, copied from the GPU host and gitignored
# (docs/REPRODUCIBILITY.md section 7): tau, trace counts and the adapter recipe per cell
# (PROV-3). The cells' filter manifests hold the counts it read; the analysis reads those.
cells:
	$(PYTHON) rebuttal/analyses/cell_recipe.py

# The returned rater files carry names and the answer key unblinds the sheets, so both
# stay outside the repo; pass their paths, e.g.
# make kappa RATER_FILES="/path/to/Grading-<rater>.xlsx" ANSWER_KEY=/path/to/answer_key.xlsx
kappa:
	@test -n '$(RATER_FILES)' || { echo "set RATER_FILES to the returned Grading-*.xlsx path(s)"; exit 1; }
	@test -n '$(ANSWER_KEY)' || { echo "set ANSWER_KEY to the answer_key.xlsx kept outside the repo"; exit 1; }
	$(PYTHON) scripts/compute_irr_kappa.py --rater-files $(RATER_FILES) --answer-key "$(ANSWER_KEY)" --filter-traces "$(FILTER_TRAIN)"

derived:
	$(PYTHON) scripts/export_derived_inputs.py --filter-train "$(FILTER_TRAIN)"
