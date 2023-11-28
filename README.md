# GPU-Accelerated-KNN
(In progress) Implemention of the KNearestNeighbors algorithm, accelerated with CUDA C++ and Python with Numba.

## Performance Improvements
insert a plot showing speedups here 

## Before Starting
Important notes:
- The code was tested on machines running Ubuntu with nvcc version {???} and CUDA version {???}.
- The python code was developed using python 3.11.
- You will need to have a machine with an NVIDIA GPU, and nvcc installed for the CUDA code to compile.

## Usage
You can use the accelerated KNN implementation via the C++ interface or the python interface.

Be sure to clone this repository or download the zip before starting.

### Python interface (`python/` directory)
1. Install the dependencies (numba, numpy, etc.): `pip install -r requirements.txt`
2. Drop the `gpu_kneighbors.py` into the project you'd like to use it in.
3. Here's how `GpuKNeighbors` might be used, assuming you read data in from a csv:

```
>>> import pandas as pd
>>> from gpu_kneighbors import GpuKNeighbors

>>> # data should be a normalized, numerical only numpy array.
>>> X = pandas.read_csv('data.csv').to_numpy()

>>> # initialize the learner to find the 5 nearest neighbors with the data
>>> gpu_knn = GpuKNeighbors(k=5)
>>> gpu_knn.fit(X)

>>> # get the distances and indices of kneighbors for the last element of X
>>> distances, indices = gpu_knn.predict(X[-1])
```

### CUDA C++ interface (`cuda/` directory)
TODO
