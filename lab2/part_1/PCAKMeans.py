import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from typing import Callable


kernel_functions = {
    "linear": lambda x, y: np.dot(x, y),
    "poly": lambda x, y, p=3: (np.dot(x, y) + 1) ** p,
    "rbf": lambda x, y, gamma=0.1: np.exp(-gamma * np.linalg.norm(x - y) ** 2),
}


def get_kernel_function(kernel: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Returns the kernel function based on the given kernel type.

    Parameters:
    kernel (str): The type of kernel ('linear', 'poly', 'rbf').

    Returns:
    function: The kernel function.
    """
    return kernel_functions[kernel]


class PCA:
    def __init__(self, n_components: int = 2, kernel: str = "linear") -> None:
        """
        Initializes the PCA object.

        Parameters:
        n_components (int): Number of components to keep.
        kernel (str): Type of kernel to use ('linear', 'poly', 'rbf').
        """
        self.n_components = n_components
        self.kernel = kernel
        self.kernel_f = get_kernel_function(kernel)
        self.X_fit = None
        self.eig_vecs = None
        self.eig_vals = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fits the PCA model on the given data.

        Parameters:
        X (np.ndarray): The data matrix with shape (n_samples, n_features).
        """
        self.X_fit = X
        if self.kernel == "linear":
            # Center the data
            X_centered = X - np.mean(X, axis=0)
            # Compute the covariance matrix
            covariance_matrix = np.dot(X_centered.T, X_centered)
        else:
            # Compute the kernel matrix
            n_samples = X.shape[0]
            K = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    K[i, j] = self.kernel_f(X[i], X[j])
            # Center the kernel matrix
            X_centered = K - K.mean(axis=0) - K.mean(axis=1).reshape(-1, 1) + K.mean()
            covariance_matrix = X_centered

        # Compute eigenvalues and eigenvectors
        eig_vals, eig_vecs = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eig_vals)[::-1]
        self.eig_vals = eig_vals[sorted_indices][: self.n_components]
        self.eig_vecs = eig_vecs[:, sorted_indices][:, : self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the data to the new basis.

        Parameters:
        X (np.ndarray): The data matrix with shape (n_samples, n_features).

        Returns:
        np.ndarray: The transformed data with shape (n_samples, n_components).
        """
        if self.kernel == "linear":
            # Center the data
            X_centered = X - np.mean(self.X_fit, axis=0)
            # Project the data
            return np.dot(X_centered, self.eig_vecs)
        else:
            # Compute the kernel matrix
            n_samples = X.shape[0]
            K = np.zeros((n_samples, self.X_fit.shape[0]))
            for i in range(n_samples):
                for j in range(self.X_fit.shape[0]):
                    K[i, j] = self.kernel_f(X[i], self.X_fit[j])
            # Center the kernel matrix
            K_centered = K - K.mean(axis=0) - K.mean(axis=1).reshape(-1, 1) + K.mean()
            # Project the data
            return np.dot(K_centered, self.eig_vecs)


class KMeans:
    def __init__(self, n_clusters: int = 3, max_iter: int = 10) -> None:
        """
        Initializes the KMeans object.

        Parameters:
        n_clusters (int): The number of clusters.
        max_iter (int): The maximum number of iterations.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = None
        self.labels = None

    def initialize_centers(self, points) -> np.ndarray:
        """
        Initializes the cluster centers.

        Parameters:
        points (np.ndarray): The data points with shape (n_samples, n_features).

        Returns:
        np.ndarray: The initialized cluster centers.
        """
        n, d = points.shape
        self.centers = np.zeros((self.n_clusters, d))
        for k in range(self.n_clusters):
            random_index = np.random.choice(n, size=10, replace=False)
            self.centers[k] = points[random_index].mean(axis=0)
        return self.centers

    def assign_points(self, points) -> np.ndarray:
        """
        Assigns each point to the nearest cluster center.

        Parameters:
        points (np.ndarray): The data points with shape (n_samples, n_features).

        Returns:
        np.ndarray: The cluster labels for each point.
        """
        n_samples = points.shape[0]
        self.labels = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            distances = np.linalg.norm(points[i] - self.centers, axis=1)
            self.labels[i] = np.argmin(distances)
        return self.labels

    def update_centers(self, points) -> None:
        """
        Updates the cluster centers based on the assigned points.

        Parameters:
        points (np.ndarray): The data points with shape (n_samples, n_features).
        """
        for k in range(self.n_clusters):
            cluster_points = points[self.labels == k]
            if len(cluster_points) > 0:
                self.centers[k] = cluster_points.mean(axis=0)

    def fit(self, points) -> None:
        """
        Fits the KMeans model on the given data.

        Parameters:
        points (np.ndarray): The data points with shape (n_samples, n_features).
        """
        self.initialize_centers(points)
        for _ in range(self.max_iter):
            self.assign_points(points)
            old_centers = self.centers.copy()
            self.update_centers(points)
            if np.all(old_centers == self.centers):
                break

    def predict(self, points) -> np.ndarray:
        """
        Predicts the closest cluster for each point.

        Parameters:
        points (np.ndarray): The data points with shape (n_samples, n_features).

        Returns:
        np.ndarray: The cluster labels for each point.
        """
        return self.assign_points(points)


def load_data() -> tuple[list, np.ndarray]:
    """
    Loads the word vectors from the pre-trained Word2Vec model.

    Returns:
    tuple: A tuple containing the list of words and the corresponding word vectors.
    """
    words = [
        "computer", "laptop", "minicomputers", "PC", "software", "Macbook", "king", "queen", "monarch", "prince",
        "ruler", "princes", "kingdom", "royal", "man", "woman", "boy", "teenager", "girl", "robber", "guy", "person",
        "gentleman", "banana", "pineapple", "mango", "papaya", "coconut", "potato", "melon", "shanghai", "HongKong",
        "chinese", "Xiamen", "beijing", "Guilin", "disease", "infection", "cancer", "illness", "twitter", "facebook",
        "chat", "hashtag", "link", "internet"
    ]
    w2v = KeyedVectors.load_word2vec_format(
        "./data/GoogleNews-vectors-negative300.bin", binary=True
    )
    vectors = [w2v[w].reshape(1, 300) for w in words]
    vectors = np.concatenate(vectors, axis=0)
    return words, vectors


if __name__ == "__main__":
    words, data = load_data()

    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(data)
    data_pca = pca.transform(data)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=7)
    kmeans.fit(data_pca)
    clusters = kmeans.predict(data_pca)

    # Plot the data
    plt.figure()
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters)
    for i in range(len(words)):
        plt.annotate(words[i], data_pca[i, :])
    plt.title("Your student ID")
    plt.savefig("PCA_KMeans.png")
