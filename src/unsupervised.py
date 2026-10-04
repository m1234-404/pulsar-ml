"""Unsupervised experiments on the pulsar (positive) class: K-means vs SOM, PCA for plots."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from .som import SOM


def run(X_pos, out="results", k=3, som_epochs=30):
    Z = StandardScaler().fit_transform(X_pos)   
    res = {"kmeans_k_sweep": {}}
    for kk in range(2, 8):                     
        lab = KMeans(kk, n_init=10, random_state=0).fit_predict(Z)
        res["kmeans_k_sweep"][kk] = float(silhouette_score(Z, lab))
    km_lab = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
    res["kmeans"] = dict(k=k, silhouette=float(silhouette_score(Z, km_lab)), sizes=np.bincount(km_lab).tolist())

    som = SOM(5, 5, epochs=som_epochs).fit(Z)
    cells = som.predict(Z)
    used = np.unique(cells)
    proto_lab = KMeans(k, n_init=10, random_state=0).fit_predict(som.W)
    som_lab = proto_lab[cells]
    res["som"] = dict(non_empty_cells=int(len(used)),
                      silhouette_all_cells=float(silhouette_score(Z, np.searchsorted(used, cells))) if len(used) > 1 else None,
                      silhouette_k_groups=float(silhouette_score(Z, som_lab)), sizes=np.bincount(som_lab).tolist())

    P = PCA(2).fit(Z)          
    Y = P.transform(Z)
    res["pca_explained_variance"] = P.explained_variance_ratio_.tolist()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(Y[:, 0], Y[:, 1], c=km_lab, s=6, cmap="viridis"); ax[0].set_title("K-means (PCA view)")
    ax[1].scatter(Y[:, 0], Y[:, 1], c=som_lab, s=6, cmap="viridis"); ax[1].set_title("SOM (PCA view)")
    for a in ax: a.set(xlabel="PC1", ylabel="PC2")
    plt.tight_layout(); plt.savefig(f"{out}/unsupervised.png", dpi=150); plt.close()
    return res
