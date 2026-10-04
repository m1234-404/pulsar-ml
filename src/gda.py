"""Gaussian Discriminant Analysis (shared covariance) implemented from scratch."""
import numpy as np


class GDA:
    def fit(self, X, y):
        self.phi = y.mean()
        self.mu0, self.mu1 = X[y == 0].mean(0), X[y == 1].mean(0)
        mu = np.where(y[:, None] == 1, self.mu1, self.mu0)
        D = X - mu
        self.sigma_inv = np.linalg.pinv(D.T @ D / len(X))
        return self

    def predict_proba(self, X):
        w = self.sigma_inv @ (self.mu1 - self.mu0)
        b = (-0.5 * (self.mu1 @ self.sigma_inv @ self.mu1 - self.mu0 @ self.sigma_inv @ self.mu0)
             + np.log(self.phi / (1 - self.phi)))
        z = np.clip(X @ w + b, -50, 50)
        return 1 / (1 + np.exp(-z))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
