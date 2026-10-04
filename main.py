"""Run the whole pipeline:  python main.py [--data data/HTRU_2.csv] [--synthetic]"""
import argparse, json, os
from src.data import load_htru2, make_synthetic, split
from src import supervised, unsupervised

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/HTRU_2.csv")
ap.add_argument("--synthetic", action="store_true", help="smoke-test on fake data (no download)")
ap.add_argument("--scratch-trees", type=int, default=30)
ap.add_argument("--som-epochs", type=int, default=30)
a = ap.parse_args()

os.makedirs("results", exist_ok=True)
X, y = make_synthetic() if a.synthetic else load_htru2(a.data)
print(f"Loaded {len(y)} samples, {y.sum()} pulsars")
Xtr, Xte, ytr, yte = split(X, y)
results = {"supervised": supervised.run(Xtr, ytr, Xte, yte, scratch_trees=a.scratch_trees),
           "unsupervised": unsupervised.run(X[y == 1], som_epochs=a.som_epochs)}
json.dump(results, open("results/results.json", "w"), indent=2)
print(json.dumps(results, indent=2))
