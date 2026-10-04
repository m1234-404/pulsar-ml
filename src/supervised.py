"""Supervised experiments: GDA baseline vs Random Forest (sklearn + scratch)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score,
                             roc_curve, precision_recall_curve, roc_auc_score)
from .data import oversample, split
from .gda import GDA
from .rf_scratch import RandomForestScratch


def proba(m, X):
    """Probability of the pulsar class for sklearn or scratch models."""
    return m.predict_proba(X)[:, 1] if hasattr(m, "classes_") else m.predict_proba(X)


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, pred), recall=recall_score(y, pred),
                precision=precision_score(y, pred, zero_division=0), f1=f1_score(y, pred),
                auc=roc_auc_score(y, p))


def cv_scores(make_model, X, y, thr=0.5, k=5):
    """K-fold CV; oversampling happens inside each fold on training data only."""
    rows = []
    for tr, va in StratifiedKFold(k, shuffle=True, random_state=0).split(X, y):
        m = make_model().fit(*oversample(X[tr], y[tr]))
        rows.append(metrics(y[va], proba(m, X[va]), thr))
    return {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}


def best_threshold(y, p):
    """Threshold maximising TPR - FPR (Youden's J), found on validation data."""
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def run(Xtr, ytr, Xte, yte, out="results", scratch_trees=30):
    res = {}
    res["gda_cv"] = cv_scores(GDA, Xtr, ytr)
    Xo, yo = oversample(Xtr, ytr)
    gda = GDA().fit(Xo, yo)
    res["gda_test"] = metrics(yte, gda.predict_proba(Xte))

    grid = dict(n_estimators=[100, 200, 300], min_samples_split=[2, 5, 10],
                max_features=[2, 3, 4, "sqrt"], max_depth=[10, 30, 60, 110, None])
    search = RandomizedSearchCV(RandomForestClassifier(random_state=0, n_jobs=-1), grid, n_iter=10,
                                scoring="recall", cv=3, random_state=0, n_jobs=-1)
    search.fit(Xo, yo)
    res["rf_best_params"] = search.best_params_
    mk = lambda: RandomForestClassifier(random_state=0, n_jobs=-1, **search.best_params_)
    res["rf_cv_thr0.5"] = cv_scores(mk, Xtr, ytr, 0.5)
    rf = mk().fit(Xo, yo)

    Xa, Xv, ya, yv = split(Xtr, ytr, test_size=0.25, seed=1)
    thr = best_threshold(yv, proba(mk().fit(*oversample(Xa, ya)), Xv))
    res["rf_threshold"] = thr
    p_te = proba(rf, Xte)
    res["rf_test_thr0.5"] = metrics(yte, p_te, 0.5)
    res["rf_test_tuned_thr"] = metrics(yte, p_te, thr)

    rfs = RandomForestScratch(n_estimators=scratch_trees, max_depth=12, max_features=3).fit(Xo, yo)
    res["rf_scratch_test"] = metrics(yte, rfs.predict_proba(Xte), 0.5)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name, p in [("GDA", gda.predict_proba(Xte)), ("RF (sklearn)", p_te),
                    ("RF (scratch)", rfs.predict_proba(Xte))]:
        f, t, _ = roc_curve(yte, p); ax[0].plot(f, t, label=name)
        pr, rc, _ = precision_recall_curve(yte, p); ax[1].plot(rc, pr, label=name)
    ax[0].set(title="ROC", xlabel="FPR", ylabel="TPR")
    ax[1].set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
    ths = np.linspace(0.01, 0.99, 50)
    ax[2].plot(ths, [recall_score(yte, p_te >= t) for t in ths], label="Recall")
    ax[2].plot(ths, [precision_score(yte, p_te >= t, zero_division=0) for t in ths], label="Precision")
    ax[2].axvline(thr, ls="--", c="gray", label="chosen thr"); ax[2].set(title="RF: threshold sweep", xlabel="threshold")
    for a in ax: a.legend()
    plt.tight_layout(); plt.savefig(f"{out}/supervised.png", dpi=150); plt.close()
    return res
