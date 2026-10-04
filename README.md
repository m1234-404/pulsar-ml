# Pulsar Candidate Classification & Clustering (HTRU2)

UE24CS352A – Machine Learning mini-project  
Team: **O P Medha** (PES2UG24CS332) · **P Dhanush** (PES2UG24CS333)

Detects real radio pulsars among noise / RFI candidates (supervised) and groups the
true pulsars into clusters (unsupervised) using the HTRU2 dataset (17,898 candidates, 1,639 pulsars, 8 features).

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
Because pulsars are rare and missing one is costly, we optimise **recall** and tune the decision threshold on a validation split (never on the test set).

## Setup
```bash
git clone <https://github.com/m1234-404/pulsar-ml> && cd pulsar-ml
python -m venv venv && source venv/bin/activate     
pip install -r requirements.txt
```
Download **HTRU2** from the UCI repository (https://archive.ics.uci.edu/dataset/372/htru2),
unzip, and place `HTRU_2.csv` in `data/`.

## Run
```bash
python main.py                      
python main.py --synthetic           
python main.py --scratch-trees 50 --som-epochs 100  
```
Outputs: `results/results.json` (all metrics), `results/supervised.png` (ROC, PR, threshold sweep),
`results/unsupervised.png` (K-means and SOM clusters in PCA space).

## Structure
```
main.py            entry point
src/data.py        loading, stratified split, oversampling
src/gda.py         GDA from scratch
src/rf_scratch.py  decision tree + random forest from scratch
src/som.py         SOM from scratch
src/supervised.py  CV, hyper-parameter search, threshold tuning, plots
src/unsupervised.py K-means, SOM, silhouette, PCA plot
```
