import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 256

@cuda.jit(device=True)
def _euclidean(query: np.ndarray, point: np.ndarray) -> float:
    """
    GPU device function for euclidean distance calculation between two data points. 
    """
    # TODO: make _euclidean a device function called from a kernel
    # placeholder, will need to change this to work w/ kernel
    return np.sqrt(np.sum((query - point) ** 2))

@cuda.jit(device=True)
def _manhattan(query: np.ndarray, point: np.ndarray) -> float:
    """
    GPU device function for euclidean distance calculation between two data points. 
    """
    # TODO: make _euclidean a device function called from a kernel
    return np.sum(np.abs(query - point))

@cuda.jit
def _knn_kernel() -> None:
    """
    Calculate the knn distances and indices on the GPU.
    """
    # TODO
    pass

class GpuKNeighbors:
    def __init__(self, k: int, dist_metric: str = 'euclidean') -> None:
        self.X = None
        self.y = None
        self.dist_metric = dist_metric

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        """
        Fit this KNN classifier with the data.
        """
        self.X = X
        self.y = y

    def predict(self, query: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Find the k nearest neighbors. Compare a single query point to every point in the dataset.

        Params:
            - query (np.ndarray): the point we want the k neighbors for (the song the user input).

        Returns:
            - tuple(distances, indices) for the k neighbors for each point.
        """
        # TODO, call the kernel from here somehow after allocating device arrays for distances, indices.
        pass