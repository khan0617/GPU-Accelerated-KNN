# Compiler settings
CXX = g++
NVCC = nvcc
# NVCC_FLAGS = -I/usr/local/cuda/include -gencode=arch=compute_60,code=\"sm_60\"
LD_FLAGS = -lcudart -L/usr/local/cuda/lib64

# If "dbg" is defined, add debugging flags, else optimize
ifdef dbg
	NVCC_FLAGS += -g -G
	CXXFLAGS += -g
else
	NVCC_FLAGS += -O2
	CXXFLAGS += -O2
endif

# Executable and objects
EXE = gpu_knn
OBJ = main.o
# CUDA_OBJ = <cuda-objects>.o  # Add CUDA object files here later

# Default target
default: $(EXE)

# Main target
main.o: main.cpp
	$(CXX) -c -o $@ main.cpp $(CXXFLAGS)

# Example CUDA object file compilation
# <cuda-objects>.o: <cuda-source>.cu <cuda-headers>.h
#	$(NVCC) -c -o $@ <cuda-source>.cu $(NVCC_FLAGS)

# Linking
$(EXE): $(OBJ) # $(CUDA_OBJ)
	$(CXX) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)

.PHONY: default clean
