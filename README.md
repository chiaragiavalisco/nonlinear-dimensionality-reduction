# Nonlinear Dimensionality Reduction Techniques: Kernel PCA & Isomap

[![MATLAB](https://img.shields.io/badge/Language-MATLAB-orange.svg)](#)
[![Course](https://img.shields.io/badge/Course-Numerical%20Methods%20for%20Data%20Mining-blue.svg)](#)

A MATLAB implementation and comparative study of linear and nonlinear manifold learning techniques, developed for the course **Numerical Methods for Data Mining** at *Università degli Studi di Napoli Federico II* (Department of Mathematics and Applications "Renato Caccioppoli").

---

## 📌 Overview

Linear dimensionality reduction techniques (e.g., standard Principal Component Analysis (PCA) and classical Multidimensional Scaling (MDS)) assume that data lie on or near a linear subspace. However, many real-world datasets exhibit intrinsic nonlinear manifold geometry, where Euclidean metrics fail to capture the true topological relationships.

This repository compares **classical MDS** against two prominent nonlinear manifold learning algorithms:
1. **Kernel Principal Component Analysis (Kernel PCA)**: Projects data into a higher (or infinite) dimensional reproducing kernel Hilbert space (RKHS) via the *kernel trick* where points become linearly separable, performing linear PCA implicitly.
2. **Isomap (Isometric Feature Mapping)**: Preserves intrinsic geodesic distances on the manifold by constructing a neighborhood graph and approximating geodesic paths before applying classical scaling.

---

## 📊 Implemented Methods

### 1. Classical Multidimensional Scaling (MDS)
* **Goal**: Reconstruct coordinates in lower dimension $k$ that best preserve pairwise Euclidean distances.
* **Pipeline**:
  1. Computes pairwise Euclidean distance matrix $D \in \mathbb{R}^{n \times n}$.
  2. Applies double centering: $B = -\frac{1}{2} H D^2 H^T$, with centering matrix $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^T$.
  3. Decomposes $B$ using SVD / Eigenvalue decomposition to obtain low-dimensional coordinates.

### 2. Kernel PCA
* **Kernel Function**: Radial Basis Function / Gaussian kernel:
  $$\kappa(x_i, x_j) = \exp\left(-\frac{\Vert{}x_i - x_j\Vert{}^2}{2\sigma^2}\right)$$
* **Pipeline**:
  1. Construct Gram matrix $K \in \mathbb{R}^{n \times n}$.
  2. Center Gram matrix: $\widetilde{K} = K - \mathbf{I}_n K - K \mathbf{I}_n + \mathbf{I}_n K \mathbf{I}_n$, where $\mathbf{I}_n = \frac{1}{n} \mathbf{1}\mathbf{1}^T$.
  3. Solve eigenvalue problem $\widetilde{K} a_k = \lambda_k (n-1) a_k$.
  4. Project data onto principal components: $y_k(x) = \frac{v_k^T K}{(n-1)\lambda_k}$.

### 3. Isomap
* **Pipeline** (Tenenbaum et al., 2000):
  1. Construct $k$-nearest neighbors graph $G = (V, E)$ based on local Euclidean distances.
  2. Estimate all-pairs geodesic distances $D_G$ using shortest-path graph traversal (Floyd-Warshall / Dijkstra).
  3. Apply classical MDS to the graph distance matrix $D_G$.
  4. Monitor the *residual variance* across dimensions $d \in [1, 10]$ to identify intrinsic dimensionality.

---

## 🧪 Benchmark Datasets & Results

The repository evaluates these methods across 4 benchmark manifolds:

| Dataset | Type | Dimension | Description / Best Method |
| :--- | :---: | :---: | :--- |
| **Two Concentric Circles** | Synthetic 2D | $2 \to 1, 2$ | $N=1200$, $\sigma=26.8$. Kernel PCA and 2D MDS linearly separate both concentric bands. |
| **Fireworks** | Synthetic 2D | $2 \to 1, 2$ | $N=1500$ (3 clusters). Kernel PCA ($\sigma=0.4, 0.8$) completely separates the 3 clusters without overlap. |
| **Swiss Roll** | Nonlinear Manifold 3D | $3 \to 1, 2$ | $N=2048$, $\sigma=15.8$. MDS and KPCA fail to unroll the manifold; **Isomap** successfully flattens it into a 2D strip. |
| **S-Curve** | Nonlinear Manifold 3D | $3 \to 1, 2$ | $N=600$, $\sigma=2.2$. Classical MDS and KPCA produce overlapping folds; **Isomap** correctly recovers the 2D plane. |

---

## 🚀 Getting Started
Prerequisites
MATLAB R2020b or later (Statistics and Machine Learning Toolbox recommended for pdist and squareform).

Running the Experiments
Clone the repository and run any experiment directly in MATLAB:


---

## 🔍 Key Findings & Theoretical Insights

* **MDS & Linear Methods**: Preserve global Euclidean distances. When data live on a nonlinear sub-manifold (like the Swiss Roll or S-curve), Euclidean short-circuits across folds cause severe projection overlaps.
* **Kernel PCA**: Avoids non-convex optimization and extracts nonlinear features by implicit mapping into an RKHS. Highly effective for cluster separation (e.g. concentric circles, fireworks), but very sensitive to the hyperparameter $\sigma$.
* **Isomap**: Excels at manifold unwrapping (Swiss Roll, S-Curve) because geodesic graph distances reflect intrinsic topology. However, it is computationally intensive ($O(N^3)$ with Floyd-Warshall) and sensitive to topological noise ("short-circuit" edges caused by poor $k$-neighbor selection or non-convex holes).

