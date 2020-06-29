
.PHONY: help summarize_repetitions setup_environment prepare_datasets datasets repetition_results_latex repetition_results_csv

HYPERPARAMETER_SEARCH := results/hyperparameter_search_balanced
BEST_RUNS := results/best_runs_balanced
REPETITIONS := results/repetitions_balanced
SUMMARY := results/summary_balanced

help:
	@# Got this from https://gist.github.com/rcmachado/af3db315e31383502660#gistcomment-1585632
	$(info Available targets)
	@awk '/^[a-zA-Z\-\_0-9]+:/ {                    \
	  nb = sub( /^## /, "", helpMsg );              \
	  if(nb == 0) {                                 \
	    helpMsg = $$0;                              \
	    nb = sub( /^[^:]*:.* ## /, "", helpMsg );   \
	  }                                             \
	  if (nb)                                       \
	    print  $$1 helpMsg;                         \
	}                                               \
	{ helpMsg = $$0 }'                              \
	$(MAKEFILE_LIST) | column -ts:

## Setup a new virtual environment using poetry, set DEVICE accordingly (DEVICE=gpu or DEVICE=cpu)
setup_environment: .environment_is_setup

.environment_is_setup:
	poetry install
	touch $@

prepare_datasets:
	poetry run seft_prepare_datasets physionet2012 physionet2019

datasets: .environment_is_setup download_datasets

## Generate csv summary table from repetitions
repetition_results_csv: $(SUMMARY)/online.csv $(SUMMARY)/binary.csv
## Generate latex summary table from repetitions
repetition_results_latex: $(SUMMARY)/online.tex $(SUMMARY)/binary.tex

$(REPETITIONS).csv: $(shell find $(REPETITIONS) -name results.json) scripts/aggregate_results.py
	poetry run python scripts/aggregate_results.py $(REPETITIONS) --output $@

## Summarize repition runs using mean and stddev
summarize_repetitions: $(REPETITIONS).csv

$(SUMMARY)/binary.csv: $(REPETITIONS).csv scripts/summarize_repetitions.py
	-mkdir -p $(SUMMARY)
	poetry run python scripts/summarize_repetitions.py $< --filter 'dataset != "physionet2019"' --csv-output $@ --all-columns --summarize-cols test_acc test_auprc test_auroc mean_epoch_time

$(SUMMARY)/binary.tex: $(REPETITIONS).csv scripts/summarize_repetitions.py
	-mkdir -p $(SUMMARY)
	poetry run python scripts/summarize_repetitions.py $< --filter 'dataset != "physionet2019"' --latex-output $@ --summarize-cols test_acc test_auprc test_auroc mean_epoch_time

$(SUMMARY)/online.csv: $(REPETITIONS).csv scripts/summarize_repetitions.py
	-mkdir -p $(SUMMARY)
	poetry run python scripts/summarize_repetitions.py $< --filter 'dataset == "physionet2019"' --csv-output $@ --all-columns --summarize-cols test_acc test_balanced_accuracy test_auprc test_auroc test_physionet2019_utility mean_epoch_time

$(SUMMARY)/online.tex: $(REPETITIONS).csv scripts/summarize_repetitions.py
	-mkdir -p $(SUMMARY)
	poetry run python scripts/summarize_repetitions.py $< --filter 'dataset == "physionet2019"' --latex-output $@ --summarize-cols test_acc test_balanced_accuracy test_auprc test_auroc test_physionet2019_utility mean_epoch_time


## Summarize the runs in the results/hyperparameter_search folder
summarize_hypersearch: $(HYPERPARAMETER_SEARCH).csv

$(HYPERPARAMETER_SEARCH).csv: $(shell find $(HYPERPARAMETER_SEARCH) -name results.json) scripts/aggregate_results.py
	poetry run python scripts/aggregate_results.py $(HYPERPARAMETER_SEARCH) --output $@

## Extract the best runs in terms of validation loss
extract_best_runs: $(BEST_RUNS).csv

$(BEST_RUNS).csv: $(HYPERPARAMETER_SEARCH).csv scripts/extract_best_runs.py
	-mkdir -p $(BEST_RUNS)
	poetry run python scripts/extract_best_runs.py $< --output_best_runs $@  --output_hyperparameters $(BEST_RUNS) --evaluation_metric best_val_auprc --selection_criterion max

