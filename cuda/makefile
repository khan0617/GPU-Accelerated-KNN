NVCC        = nvcc
NVCC_FLAGS  = -I/usr/local/cuda/include -gencode=arch=compute_60,code=\"sm_60\" -std=c++11
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O3
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = gpu_knn
OBJ	        = main.o knn.o
HEADERS		= support.h knn.h

default: $(EXE)

main.o: main.cpp knn.cu $(HEADERS)
	$(NVCC) -c -o $@ main.cpp $(NVCC_FLAGS)

knn.o: knn.cu $(HEADERS)
	$(NVCC) -c -o $@ knn.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
