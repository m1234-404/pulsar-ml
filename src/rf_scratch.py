"""Random Forest implemented from scratch (entropy / information gain)."""
import numpy as np


def _entropy(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    p = p[p > 0]
    return -(p * np.log2(p)).sum()


class _Node:
    __slots__ = ("feat", "thr", "left", "right", "p")


class DecisionTree:
    def __init__(self, max_depth=12, min_samples_split=5, max_features=3, n_thresholds=16, rng=None):
        self.max_depth, self.min_split = max_depth, min_samples_split
        self.max_features, self.n_thr = max_features, n_thresholds
        self.rng = rng or np.random.default_rng()

    def fit(self, X, y):
        self.root = self._build(X, y, 0)
        return self

    def _best_split(self, X, y):
        parent, n = _entropy(y), len(y)
        best = (0.0, None, None)
        for f in self.rng.choice(X.shape[1], self.max_features, replace=False):
            thrs = np.unique(np.percentile(X[:, f], np.linspace(5, 95, self.n_thr)))
            for t in thrs:
                m = X[:, f] <= t
                nl = m.sum()
                if nl == 0 or nl == n:
                    continue
                gain = parent - nl / n * _entropy(y[m]) - (n - nl) / n * _entropy(y[~m])
                if gain > best[0]:
                    best = (gain, f, t)
        return best

    def _build(self, X, y, depth):
        node = _Node()
        node.p = y.mean()
        node.feat = node.left = node.right = None
        if depth >= self.max_depth or len(y) < self.min_split or node.p in (0.0, 1.0):
            return node
        gain, f, t = self._best_split(X, y)
        if f is None:
            return node
        m = X[:, f] <= t
        node.feat, node.thr = f, t
        node.left, node.right = self._build(X[m], y[m], depth + 1), self._build(X[~m], y[~m], depth + 1)
        return node

    def predict_proba(self, X):
        out = np.empty(len(X))
        for i, x in enumerate(X):
            n = self.root
            while n.feat is not None:
                n = n.left if x[n.feat] <= n.thr else n.right
            out[i] = n.p
        return out


class RandomForestScratch:
    def __init__(self, n_estimators=50, max_depth=12, min_samples_split=5, max_features=3, seed=0):
        self.n, self.seed = n_estimators, seed
        self.kw = dict(max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for _ in range(self.n):
            idx = rng.integers(0, len(X), len(X))  # bootstrap sample
            self.trees.append(DecisionTree(rng=rng, **self.kw).fit(X[idx], y[idx]))
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.trees], axis=0)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
