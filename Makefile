# One command from the committed results to the paper's numbers, figures and PDF.
#   make numbers   paper/generated/{numbers.tex,numbers.json,tab_*.tex}
#   make figures   paper/figures/*.pdf
#   make paper     both, then paper/build/main.pdf
#   make test      the test suite
#   make rebuttal  rebuttal/results/*.json from results_modal/
#   make kappa     results_modal/irr/kappa_results.json (needs RATER_FILES)

PYTHON ?= python3
# Inputs kept out of git. A target that needs one skips that table, with a message,
# when the file is absent; point these at the files if they live elsewhere.
HB_HARD ?= data/raw/healthbench_hard.jsonl
FILTER_TRAIN ?= data/sft_qwen_graded/seed_7_train.jsonl
POSTED_VALUES ?= data/rebuttal_cells/posted_values.json

.PHONY: numbers figures paper test rebuttal kappa

numbers:
	$(PYTHON) scripts/paper_stats.py --healthbench "$(HB_HARD)" --filter-train "$(FILTER_TRAIN)" \
		--posted-values "$(POSTED_VALUES)"

figures:
	$(PYTHON) scripts/build_paper_figures.py --hb-hard "$(HB_HARD)"

paper: numbers figures
	mkdir -p paper/build
	tectonic -X compile --outdir paper/build paper/main.tex

test:
	$(PYTHON) -m pytest -q -p no:cacheprovider

rebuttal:
	$(PYTHON) rebuttal/analyses/mimicry_split.py --healthbench "$(HB_HARD)"
	$(PYTHON) rebuttal/analyses/interference_mechanism.py
	$(PYTHON) rebuttal/analyses/second_person_questions.py

# The returned rater files carry names, so they stay outside the repo; pass their
# paths, e.g. make kappa RATER_FILES="/path/to/Grading-<rater>.xlsx".
kappa:
	@test -n '$(RATER_FILES)' || { echo "set RATER_FILES to the returned Grading-*.xlsx path(s)"; exit 1; }
	$(PYTHON) scripts/compute_irr_kappa.py --rater-files $(RATER_FILES) --filter-traces "$(FILTER_TRAIN)"
