# Pulsar Candidate Classification & Clustering (HTRU2)

UE24CS352A – Machine Learning mini-project  
Team: **O P Medha** (PES2UG24CS332) · **P Dhanush** (PES2UG24CS333)

Detects real radio pulsars among noise / RFI candidates (supervised) and groups the
true pulsars into clusters (unsupervised) using the HTRU2 dataset
(17,898 candidates, 1,639 pulsars, 8 features).

## What is implemented
| Part | Method | Notes |
|---|---|---|
| Baseline classifier | Gaussian Discriminant Analysis | from scratch (NumPy) |
| Main classifier | Random Forest (scikit-learn) | `RandomizedSearchCV`, recall-driven threshold tuning |
| Random Forest | from scratch | entropy / information gain, bootstrap, random feature subsets |
| Baseline clustering | K-means (k = 3) | scikit-learn, silhouette sweep for k = 2..7 |
| Clustering | Self-Organizing Map (5x5) | from scratch (NumPy) |
| Visualisation | PCA to 2-D | used only for plotting, not for training |

Imbalance handling: random oversampling of pulsars, applied **only to training data** (inside each CV fold).
Because pulsars are rare and missing one is costly, we optimise **recall** and tune the decision threshold on a
validation split of the training data (never on the test set).

## Setup
```bash
git clone https://github.com/m1234-404/pulsar-ml.git
cd pulsar-ml
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
python -m pip install -r requirements.txt
```
Requires Python 3.9+ (tested on 3.12).

### Dataset
Download **HTRU2** from the UCI repository (https://archive.ics.uci.edu/dataset/372/htru2),
unzip it, and place **`HTRU_2.csv`** in the `data/` folder, so the path is `data/HTRU_2.csv`.
The CSV has no header row: 8 feature columns followed by the label (1 = pulsar, 0 = not).
The dataset is not stored in the repository.

## Run
Run all commands from the project root folder.
```bash
python main.py                       # full pipeline on data/HTRU_2.csv (about 2-5 minutes)
python main.py --synthetic           # quick smoke test on fake data (no download needed)
python main.py --scratch-trees 50 --som-epochs 100   # slower, heavier settings
```
Command-line options: `--data PATH`, `--synthetic`, `--scratch-trees N` (default 30), `--som-epochs N` (default 30).

> `--synthetic` uses made-up data only to check that the code runs. Its metrics are meaningless;
> use the real-data run for results.

## Outputs
Written to `results/`:
- `results.json` – all metrics (cross-validation and test set, clustering scores)
- `supervised.png` – ROC curve, precision-recall curve, Random Forest threshold sweep
- `unsupervised.png` – K-means and SOM clusters projected onto 2 principal components

## Results (HTRU2, 20% stratified hold-out test set)
| Model | Accuracy | Recall | Precision | F1 | AUC |
|---|---|---|---|---|---|
| GDA (baseline) | 97.91% | 89.63% | 87.76% | 88.69% | 0.971 |
| Random Forest (sklearn), thr 0.5 | 97.82% | 88.41% | 87.88% | 88.15% | 0.969 |
| Random Forest (sklearn), tuned thr 0.286 | 97.40% | 90.24% | 82.91% | 86.42% | 0.969 |
| Random Forest (from scratch), thr 0.5 | 97.82% | 89.94% | 86.76% | 88.32% | 0.973 |

| Clustering (1,639 pulsars) | Silhouette |
|---|---|
| K-means (k = 3) | 0.335 |
| SOM 5x5, grouped into 3 clusters | 0.314 |

## Project structure
```
main.py               entry point (runs supervised + unsupervised experiments)
requirements.txt      Python dependencies
src/data.py           loading, stratified split, oversampling, synthetic data
src/gda.py            Gaussian Discriminant Analysis from scratch
src/rf_scratch.py     decision tree + random forest from scratch
src/som.py            Self-Organizing Map from scratch
src/supervised.py     CV, hyper-parameter search, threshold tuning, plots
src/unsupervised.py   K-means, SOM, silhouette, PCA plot
data/                 put HTRU_2.csv here (not tracked by git)
results/              generated metrics and figures
```

## Reference
R. J. Lyon et al., *Fifty Years of Pulsar Candidate Selection*, MNRAS 459(1), 2016.
Dataset: R. J. Lyon, HTRU2, UCI Machine Learning Repository.
