# script to test out the GPU KNN implementation.
# the output is the same as the CUDA code.

import numpy as np
from gpu_kneighbors import GpuKNeighbors
X = np.genfromtxt('../cuda/normalized_knn_data.csv', delimiter=',')
gpu_knn = GpuKNeighbors(k=7)
gpu_knn.fit(X)
query_index = 45000
distances, indices = gpu_knn.predict(X[query_index])

print(f'GpuKNeighbors distances/indices for {query_index = }')
for i, (dist, idx) in enumerate(zip(distances, indices)):
    print(f'{i=}, distance={dist:.6f}, index_into_data={idx}')