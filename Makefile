# One command from the committed results to the paper's numbers, figures and PDF.
#   make numbers   paper/generated/{numbers.tex,numbers.json,tab_*.tex}
#   make figures   paper/figures/*.pdf
#   make paper     both, then paper/build/main.pdf
#   make test      the test suite
#   make rebuttal  rebuttal/results/*.json from results_modal/
#   make cells     rebuttal-cell values and paired tests (needs results_rebuttal/ synced)
#   make kappa     results_modal/irr/kappa_results.json (needs RATER_FILES)
#   make derived   data/derived/*.json from the large ignored files they copy scores from

PYTHON ?= python3
# Inputs kept out of git; point these at the files if they live elsewhere.
# HealthBench-Hard is public, and setup.sh downloads it. numbers and figures stop
# without it, because the paper uses its theme tags and rubric points.
HB_HARD ?= data/raw/healthbench_hard.jsonl
# The seed-7 training traces (49 MB). make numbers reads only their scores, from the
# tracked data/derived/seed_7_filter_scores.json; kappa and derived need the file.
FILTER_TRAIN ?= data/sft_qwen_graded/seed_7_train.jsonl
POSTED_VALUES ?= data/rebuttal_cells/posted_values.json

.PHONY: numbers figures paper test rebuttal cells kappa derived

numbers: $(HB_HARD)
	$(PYTHON) scripts/paper_stats.py --healthbench "$(HB_HARD)" --posted-values "$(POSTED_VALUES)"

figures: $(HB_HARD)
	$(PYTHON) scripts/build_paper_figures.py --hb-hard "$(HB_HARD)"

# Runs only when the file is missing: stop with the fix instead of a traceback.
$(HB_HARD):
	@echo "$(HB_HARD) not found. Run bash setup.sh (it downloads HealthBench), or pass HB_HARD=<path>."
	@exit 1

paper: numbers figures
	$(PYTHON) scripts/check_paper_macros.py
	mkdir -p paper/build
	tectonic -X compile --outdir paper/build paper/main.tex

test:
	$(PYTHON) -m pytest -q -p no:cacheprovider

rebuttal: $(HB_HARD)
	$(PYTHON) rebuttal/analyses/mimicry_split.py --healthbench "$(HB_HARD)"
	$(PYTHON) rebuttal/analyses/interference_mechanism.py
	$(PYTHON) rebuttal/analyses/second_person_questions.py

# The added cells' raw outputs are on the shared GPU host until synced (audit PROV-1).
# cell_values.py diffs them against data/rebuttal_cells/posted_values.json.
cells:
	$(PYTHON) rebuttal/analyses/cell_values.py
	$(PYTHON) rebuttal/analyses/quality_drop.py

# The returned rater files carry names, so they stay outside the repo; pass their
# paths, e.g. make kappa RATER_FILES="/path/to/Grading-<rater>.xlsx".
kappa:
	@test -n '$(RATER_FILES)' || { echo "set RATER_FILES to the returned Grading-*.xlsx path(s)"; exit 1; }
	$(PYTHON) scripts/compute_irr_kappa.py --rater-files $(RATER_FILES) --filter-traces "$(FILTER_TRAIN)"

derived:
	$(PYTHON) scripts/export_derived_inputs.py --filter-train "$(FILTER_TRAIN)"
