"""Self-Organizing Map implemented from scratch with NumPy."""
import numpy as np


class SOM:
    def __init__(self, rows=5, cols=5, epochs=100, lr0=0.5, seed=0):
        self.rows, self.cols, self.epochs, self.lr0, self.seed = rows, cols, epochs, lr0, seed

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        self.W = X[rng.choice(len(X), self.rows * self.cols)].copy()  
        self.grid = np.array([(r, c) for r in range(self.rows) for c in range(self.cols)], float)
        sigma0 = max(self.rows, self.cols) / 2
        for e in range(self.epochs):
            decay = np.exp(-e / self.epochs)
            lr, sigma = self.lr0 * decay, sigma0 * decay
            for x in X[rng.permutation(len(X))]:
                bmu = np.argmin(((self.W - x) ** 2).sum(1))         
                d2 = ((self.grid - self.grid[bmu]) ** 2).sum(1)      
                theta = np.exp(-d2 / (2 * sigma ** 2))               
                self.W += lr * theta[:, None] * (x - self.W)
        return self

    def predict(self, X):
        d = ((X[:, None, :] - self.W[None]) ** 2).sum(2)
        return d.argmin(1)
