# Graph Neural Network Dashboard for Explainable Cryptocurrency Money Laundering Detection

This repository contains the data pipeline, model training, explainability, and
figure-generation code for a dissertation project applying a Graph Attention
Network (GAT) to anti-money laundering (AML) detection on the Elliptic Bitcoin
Dataset, paired with SHAP explainability and a decision-support dashboard
evaluated by real AML analysts.

The full write-up (methodology, literature review, usability evaluation, and
discussion) is in the accompanying dissertation. This repository contains the
underlying, runnable code, the results it produced, and the decision-support
dashboard prototype that five AML analysts evaluated.

## Live dashboard

**[Open the dashboard](https://jessydaniella.github.io/gnn-aml-explainable-dashboard/)** —
a live, interactive copy of the LedgerLens dashboard shown in Chapter Four
(Figures 4.7–4.9). No installation needed; it opens directly in the browser.

## What's here

```
index.html  The dashboard prototype (served live via GitHub Pages, see link above)
dashboard/  Same dashboard file, kept here for direct download/reference
scripts/    5 pipeline scripts, run in order (see "Running the pipeline" below)
results/    Small, final result artifacts: trained model, SHAP outputs, figures, metrics
data/       Where you place the raw Elliptic Bitcoin Dataset files (not included, see below)
outputs/    Where intermediate pipeline files are written when you run the scripts
```

## Pipeline

| Script | What it does |
|---|---|
| `ch4_step1_merge_features.py` | Merges the raw Elliptic feature files into one CSV |
| `ch4_step2_build_graph.py` | Builds the PyTorch Geometric graph and applies the temporal train/validation/test split |
| `ch4_step3_train_gat.py` | Defines and trains the two-layer GAT, evaluates it on the held-out test set |
| `ch4_step4_shap_explain.py` | Generates SHAP explanations for a representative sample of test-set predictions (true positives, false positives, true negatives) |
| `ch4_step5_generate_figures.py` | Generates the training curve, ROC curve, and confusion matrix figures from the real results |

## Getting the data

This project uses the **Elliptic Bitcoin Dataset**, a public dataset of Bitcoin
transactions labelled licit/illicit, released by Elliptic and Weber et al.
(2019). It is not redistributed in this repository; download it yourself (e.g.
from Kaggle: "Elliptic Data Set") and place the following four files in `data/`:

```
data/elliptic_txs_features_batch1.xlsx   (or elliptic_txs_features.csv, see note below)
data/elliptic_txs_features_batch2.xlsx
data/elliptic_txs_classes.csv
data/elliptic_txs_edgelist.csv
```

Note: the original Elliptic release is a single `elliptic_txs_features.csv`.
This project's `ch4_step1` script expects the features split into two Excel
batches (`batch1.xlsx`, `batch2.xlsx`) as they were originally supplied for
this project; if you're starting from the standard single CSV, either split
it into two `.xlsx` batches first, or adapt `ch4_step1_merge_features.py` to
read your CSV directly (a few lines).

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. Tested with the versions in `requirements.txt`; a CPU-only
machine is sufficient (no GPU required — training takes under 3 minutes on CPU).

## Running the pipeline

Run the five scripts in order from the repository root:

```bash
python scripts/ch4_step1_merge_features.py
python scripts/ch4_step2_build_graph.py
python scripts/ch4_step3_train_gat.py
python scripts/ch4_step4_shap_explain.py
python scripts/ch4_step5_generate_figures.py
```

Each step reads the previous step's output from `outputs/` and writes its own
output there. Total runtime end-to-end is around 6-7 minutes on a standard
CPU machine (most of it is Step 1, reading the raw Excel files).

## Results

These are the actual results this pipeline produces, reproduced end-to-end
and saved in `results/`.

**Dataset** (after Step 2): 203,769 transactions, 234,355 edges, 165 features,
49 time steps. Temporal split per the dissertation's Chapter Three, Table 3.2:

| Split | Time steps | Nodes | Illicit | % illicit |
|---|---|---|---|---|
| Train | 1-34 | 29,894 | 3,462 | 11.58% |
| Validation | 35-39 | 5,486 | 447 | 8.15% |
| Test | 40-49 | 11,184 | 636 | 5.69% |

**Model**: 2-layer GAT, 4 attention heads, hidden dimension 16, class-weighted
loss (licit 1.0, illicit 7.63), Adam optimiser, 80 epochs.

**Test-set results** (held out, time steps 40-49, n = 11,184):

| Metric | Value |
|---|---|
| Accuracy | 0.893 |
| Precision | 0.284 |
| Recall | 0.585 |
| F1-score | 0.383 |
| ROC-AUC | 0.846 |
| Confusion matrix | TN=9,612, FP=936, FN=264, TP=372 |

**Key finding**: the model's four highest-confidence predictions (all >99.5%)
were false positives — none were correctly identified illicit transactions.
The highest-confidence *true* illicit prediction reached only 98.6% (ranked
5th overall by confidence). This confidence-reliability mismatch is why the
dashboard component of this project pairs every prediction with a SHAP
explanation rather than surfacing the confidence score alone — see the
dissertation, Chapter Four, Section 4.4, for the full discussion.

**SHAP explainability** (`results/ch4_shap_results.json`): explanations for 6
real test-set transactions, deliberately selected to include 2 true positives,
2 false positives, and 2 true negatives, rather than only successful cases.

**Figures** (`results/figures/`): training curve, confusion matrix heatmap,
and ROC curve, generated directly from the real training log and test-set
predictions.

## A note on reproducibility

Every number above was produced by actually running the five scripts in this
repository against the real dataset, not copied from a prior run. `SEED = 42`
is set in both `ch4_step3_train_gat.py` and `ch4_step4_shap_explain.py` for
reproducibility, though exact SHAP values may vary slightly between machines
due to KernelSHAP's sampling procedure.

## Context

This code was developed as part of an MRes/MSc dissertation on explainable
GNN-based AML detection. The trained model and SHAP explanations feed into a
decision-support dashboard (not included in this repository) that was
evaluated by five real AML analysts; see the dissertation for the dashboard,
usability evaluation, and full discussion.

## License

Code in this repository is provided for academic and research purposes. The
Elliptic Bitcoin Dataset is subject to its own license terms, available from
its original source, and is not included here.
