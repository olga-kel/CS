# Span Projection Reproducibility Package

This package reproduces the analysis workflow for **“Feature Scope and Span Projection in Bilingual Grammar: A Dependency-Based Analysis of Spanish–English and Spanish–Guaraní Code-Switching.”**

## Contents

### Data
- `data/miami_gpt_labels_sent.xlsx` — processed Miami Spanish–English corpus.
- `data/spa_gua_gpt_labels_sent-2.xlsx` — processed Paraguay Spanish–Guaraní corpus.
- `data/paraguay_loanword_mapping.csv` — explicit Paraguayan loanword relabeling file used by the script when present.

### Scripts
- `scripts/span_projection_analysis.py` — main analysis script.
- `scripts/span_projection_utils.py` — helper functions for parsing, classification, and summary statistics.

### Results
- `results/TABLE2_functional_vs_lexical.csv`
- `results/TableS2_ttests_and_effectsizes.csv`
- `results/TableS1_by_UPOS.csv`
- `results/SUPPLEMENT_Full_DEPREL_Table.csv`
- `results/regression_terms.csv`
- `results/paper_to_output_map.csv`
- `results/miami_span_observations.csv`
- `results/paraguay_span_observations.csv`
- `results/combined_span_observations.csv`
- `results/miami_sentence_level_summary.csv`
- `results/paraguay_sentence_level_summary.csv`

## What the workflow does

The script:

1. reads the two processed corpora;
2. reconstructs token-level contiguous language runs;
3. computes span observations and sentence-level switch counts;
4. applies an explicit Paraguayan loanword relabeling step if `data/paraguay_loanword_mapping.csv` is present;
5. regenerates the summary tables used in the manuscript;
6. fits sentence-level Poisson models, span-length OLS models, and token-level logistic models;
7. writes a compact map from manuscript tables to output files.

## How to run

From the folder containing the package:

```bash
python scripts/span_projection_analysis.py \
  --miami data/miami_gpt_labels_sent.xlsx \
  --paraguay data/spa_gua_gpt_labels_sent-2.xlsx \
  --loanword-map data/paraguay_loanword_mapping.csv \
  --outdir results
```

If you prefer not to apply the loanword relabeling step, omit `--loanword-map` or point it to an empty/nonexistent file.

## Interpretation of the output

- `TABLE2_functional_vs_lexical.csv` corresponds to the functional-vs-lexical span comparison.
- `TableS2_ttests_and_effectsizes.csv` contains the t-tests and effect sizes.
- `TableS1_by_UPOS.csv` summarizes spans by UPOS.
- `SUPPLEMENT_Full_DEPREL_Table.csv` summarizes spans by DEPREL.
- `regression_terms.csv` contains the coefficient tables for the Poisson, OLS, and logistic models.
- `paper_to_output_map.csv` helps reviewers connect manuscript tables to the files in `results/`.

## Loanword relabeling

The Paraguay corpus includes Spanish-origin items that are treated as entrenched borrowings in the manuscript’s analytical workflow. The optional mapping file provides an explicit, editable list of forms to be relabelled before span segmentation. The script loads this file when present and uses it to relabel matching Spanish-tagged tokens as Guaraní-side items for analytical purposes.

## Requirements

The workflow uses standard Python scientific libraries:
- pandas
- numpy
- scipy
- statsmodels
- openpyxl

A minimal installation is:

```bash
pip install pandas numpy scipy statsmodels openpyxl
```

## Notes

- The package is designed to be auditable and easy to adapt.
- The processed corpora supplied here are the inputs used for all outputs in `results/`.
- The code keeps the operationalization of span segmentation explicit so that a reviewer can inspect the assumptions directly.
