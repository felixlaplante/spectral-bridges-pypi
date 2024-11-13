import numpy as np
import faiss
from scipy.sparse.csgraph import laplacian
from scipy.linalg.blas import sgemm


class _KMeans:
    """K-means clustering using FAISS.

    Parameters:
    -----------
    n_clusters : int
        The number of clusters to form.
    n_iter : int, optional, default=20
        Number of iterations to run the k-means algorithm.
    n_local_trials : int or None, optional, default=None
        Number of seeding trials for centroids initialization.
    random_state : int or None, optional, default=None
        Determines random number generation for centroid initialization.

    Attributes:
    -----------
    cluster_centers_ : numpy.ndarray
        Coordinates of cluster centers.
    labels_ : numpy.ndarray
        Labels of each point (index) in X.

    Methods:
    --------
    fit(X):
        Run k-means clustering on the input data X.
    """

    def __init__(
        self,
        n_clusters,
        n_iter=20,
        n_local_trials=None,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.n_local_trials = n_local_trials
        self.random_state = random_state

    def _dists(self, X, y, XX):
        yy = np.einsum("ij,ij->i", y, y)
        dists = XX - sgemm(2.0, X, y, trans_b=True) + yy
        np.clip(dists, 0, None, out=dists)
        return dists

    def _init_centroids(self, X):
        rng = np.random.default_rng(self.random_state)

        centroids = np.empty((self.n_clusters, X.shape[1]), dtype=X.dtype)
        centroids[0] = X[rng.integers(X.shape[0])]

        XX = np.einsum("ij,ij->i", X, X)[:, None]

        dists = self._dists(X, centroids[0:1], XX).ravel()
        inertia = dists.sum()

        if self.n_local_trials is None:
            self.n_local_trials = 2 + int(np.log(self.n_clusters))

        for i in range(1, self.n_clusters):
            candidate_ids = rng.choice(
                X.shape[0], size=self.n_local_trials, p=dists / inertia
            )
            candidates = np.asfortranarray(X[candidate_ids])

            current_candidates_dists = self._dists(X, candidates, XX)
            candidates_dists = np.minimum(current_candidates_dists, dists[:, None])

            inertias = candidates_dists.sum(axis=0)
            best_inertia = inertias.argmin()
            best_candidate = candidate_ids[best_inertia]
            dists = candidates_dists[:, best_inertia]
            inertia = inertias[best_inertia]

            centroids[i] = X[best_candidate]

        return centroids

    def fit(self, X):
        """Run k-means clustering on the input data X.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data to cluster.
        """
        X_f32 = np.array(X, dtype=np.float32, order="F")
        index = faiss.IndexFlatL2(X.shape[1])
        kmeans = faiss.Clustering(X.shape[1], self.n_clusters)

        init_centroids = self._init_centroids(X_f32)

        kmeans.centroids.resize(init_centroids.size)
        faiss.copy_array_to_vector(init_centroids.ravel(), kmeans.centroids)
        kmeans.niter = self.n_iter
        kmeans.min_points_per_centroid = 0
        kmeans.max_points_per_centroid = -1
        kmeans.train(X_f32, index)

        self.cluster_centers_ = faiss.vector_to_array(kmeans.centroids).reshape(
            self.n_clusters, X.shape[1]
        )
        self.labels_ = index.search(X_f32, 1)[1].ravel()


class _SpectralClustering:
    """Spectral clustering based on Laplacian matrix.

    Parameters:
    -----------
    n_clusters : int
        The number of clusters to form.
    random_state : int or None, optional, default=None
        Determines random number generation for centroid initialization.

    Attributes:
    -----------
    labels_ : numpy.ndarray
        Labels of each point (index) in the affinity matrix.
    eigvals_ : numpy.ndarray
        The eigenvalues of the (normalized) laplacian

    Methods:
    --------
    fit(affinity):
        Fit the spectral clustering model on the affinity matrix.
    """

    def __init__(self, n_clusters, random_state):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_ = None
        self.eigvals_ = None

    def fit(self, affinity):
        """Fit the spectral clustering model on the affinity matrix.

        Parameters:
        -----------
        affinity : numpy.ndarray
            Affinity matrix representing pairwise similarity between points.
        """
        L = laplacian(affinity, normed=True)

        self.eigvals_, eigvecs = np.linalg.eigh(L)
        eigvecs = eigvecs[:, : self.n_clusters]
        eigvecs /= np.linalg.norm(eigvecs, axis=1)[:, None]
        kmeans = _KMeans(self.n_clusters, random_state=self.random_state)
        kmeans.fit(eigvecs)

        self.labels_ = kmeans.labels_


class SpectralBridges:
    """Spectral Bridges clustering algorithm.

    Parameters:
    -----------
    n_clusters : int
        The number of clusters to form.
    n_nodes : int
        Number of nodes or initial clusters.
    M : float, optional, default=1e4
        Scaling parameter for affinity matrix computation.
    n_iter : int, optional, default=20
        Number of iterations to run the k-means algorithm.
    n_local_trials : int or None, optional, default=None
        Number of seeding trials for centroids initialization.
    random_state : int or None, optional, default=None
        Determines random number generation for centroid initialization.

    Methods:
    --------
    fit(X):
        Fit the Spectral Bridges model on the input data X.
    predict(x):
        Predict the nearest cluster index for each input data point x.
    """

    def __init__(
        self,
        n_clusters,
        n_nodes,
        M=1e4,
        n_iter=20,
        n_local_trials=None,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.n_nodes = n_nodes
        self.M = M
        self.n_iter = n_iter
        self.n_local_trials = n_local_trials
        self.random_state = random_state
        self.cluster_centers_ = None
        self.eigvals_ = None

    def fit(self, X):
        """Fit the Spectral Bridges model on the input data X.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data to cluster.
        """
        kmeans = _KMeans(
            self.n_nodes,
            n_iter=self.n_iter,
            n_local_trials=self.n_local_trials,
            random_state=self.random_state,
        )
        kmeans.fit(X)

        affinity = np.empty((self.n_nodes, self.n_nodes))

        X_centered = [
            np.array(
                X[kmeans.labels_ == i] - kmeans.cluster_centers_[i],
                dtype=np.float32,
                order="F",
            )
            for i in range(self.n_nodes)
        ]

        counts = np.array([X_centered[i].shape[0] for i in range(self.n_nodes)])
        counts = counts[None, :] + counts[:, None]

        for i in range(self.n_nodes):
            segments = np.asfortranarray(
                kmeans.cluster_centers_ - kmeans.cluster_centers_[i]
            )
            dists = np.einsum("ij,ij->i", segments, segments)
            dists[i] = 1

            projs = sgemm(1.0, X_centered[i], segments, trans_b=True) / dists
            np.clip(projs, 0, None, out=projs)

            affinity[i] = np.einsum("ij,ij->j", projs, projs)

        affinity = np.sqrt((affinity + affinity.T) / counts)
        affinity -= 0.5 * affinity.max()

        q10, q90 = np.quantile(affinity, [0.1, 0.9])

        gamma = np.log(self.M) / (q90 - q10)
        affinity = np.exp(gamma * affinity)

        spectralclustering = _SpectralClustering(
            self.n_clusters, random_state=self.random_state
        )
        spectralclustering.fit(affinity)

        self.eigvals_ = spectralclustering.eigvals_
        self.cluster_centers_ = [
            kmeans.cluster_centers_[spectralclustering.labels_ == i]
            for i in range(self.n_clusters)
        ]

    def predict(self, x):
        """Predict the nearest cluster index for each input data point x.

        Parameters:
        -----------
        x : numpy.ndarray
            Input data points to predict clusters.

        Returns:
        --------
        numpy.ndarray
            Predicted cluster indices for each input data point.
        """
        cluster_centers = np.vstack(self.cluster_centers_)
        cluster_cutoffs = np.cumsum(
            [cluster.shape[0] for cluster in self.cluster_centers_]
        )

        index = faiss.IndexFlatL2(x.shape[1])
        index.add(cluster_centers.astype(np.float32))
        winners = index.search(x.astype(np.float32), 1)[1].ravel()

        labels = np.searchsorted(cluster_cutoffs, winners, side="right")

        return labels

    def normalized_eigengap(self):
        """Returns the normalized eigengap

        Returns:
        --------
        float
            Normalized eigengap value
        """
        return (
            self.eigvals_[self.n_clusters] - self.eigvals_[self.n_clusters - 1]
        ) / self.eigvals_[self.n_clusters]


def best_n_nodes(
    X,
    n_clusters,
    n_nodes_range,
    n_redo=10,
    M=1e4,
    n_iter=20,
    n_local_trials=None,
    random_state=None,
):
    """Finds the optimal number of nodes the Spectral Bridges model on the input data X.

    Parameters:
    -----------
    X : numpy.ndarray
        Input data to cluster.
    n_clusters : int
        The number of clusters to form.
    n_nodes_range : (list of int)
        A list of integer values representing the number of nodes to test.
    M : float, optional, default=1e4
        Scaling parameter for affinity matrix computation.
    n_iter : int, optional, default=20
        Number of iterations to run the k-means algorithm.
    n_local_trials : int or None, optional, default=None
        Number of seeding trials for centroids initialization.
    random_state : int or None, optional, default=None
        Determines random number generation for centroid initialization.

    Returns:
    --------
    tuple: A tuple containing:
        - SpectralBridges object (best fitted model)
        - int: The optimal number of nodes.
        - float: The mean value of the best normalized eigengap.
    """
    models = []
    normalized_eigengaps = np.zeros(len(n_nodes_range))
    for i, n_nodes in enumerate(n_nodes_range):
        for _ in range(n_redo):
            model = SpectralBridges(
                n_clusters=n_clusters,
                n_nodes=n_nodes,
                M=M,
                n_iter=n_iter,
                n_local_trials=n_local_trials,
                random_state=random_state,
            )
            model.fit(X)
            models.append(model)

            normalized_eigengaps[i] += model.normalized_eigengap()

            if random_state is not None:
                random_state += 1

    idx = np.argmax(normalized_eigengaps)
    best_model = models[idx]
    best_n_nodes = n_nodes_range[idx]
    best_mean_normalized_eigengap = normalized_eigengaps[idx] / n_redo
    return (best_model, best_n_nodes, best_mean_normalized_eigengap)
