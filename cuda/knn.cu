#include <stdio.h>
#include "support.h"

#define BLOCK_SIZE 256
__constant__ float query_c[NUM_DATA_FEATURES];

/**
 * Return the indices that would sort the distances array.
 * 
 * @param distance_index The array of structs which holds the distance to the query and the index of that record
 * @param size The number of values grouped for the current sorting
 * @param width The distance between distance_index entries that are being compared
*/
__global__ void bitonicsort_kernel(distance_index_t *distance_index, int size, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    bool sorting_thread = (idx % width) < (width / 2);
    int relative_idx = idx % (2*size);

    if (sorting_thread) {
        if (relative_idx < size) { //sort descending 
            if (distance_index[idx].distance > distance_index[idx + (width / 2)].distance) {
                // idx is larger than idx + w/2 so swap idx further down (descending)
                distance_index_t temp = distance_index[idx];
                distance_index[idx] = distance_index[idx + (width / 2)];
                distance_index[idx + (width / 2)] = temp;
            }
        }
        else { //sort ascending
            if (distance_index[idx].distance < distance_index[idx + (width / 2)].distance) {
                // idx is smaller than idx + w/2 so swap idx further up (ascending)
                distance_index_t temp = distance_index[idx];
                distance_index[idx] = distance_index[idx + (width / 2)];
                distance_index[idx + (width / 2)] = temp;
            }
            // here is where equivalent values will do nothing
        }
    }
    return;
}

/**
 * Calculate the euclidean distance between query (in constant memory) and every point in the dataset.
 * 
 * @param data Flattened dataset with NUM_RECORDS * NUM_DATA_FEATURES elements.
 * @param distance_index Output vector, each struct holds its distance and original index in the dataset. 
 *  ex: distances[0].distance is the distance between data[0] and the query point.
*/
__global__ void euclidean_distance_kernel(float *data, distance_index_t *distance_index) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_RECORDS) {
        float sum = 0;
        for (int i = 0; i < NUM_DATA_FEATURES; i++) {
            float diff = data[idx * NUM_DATA_FEATURES + i] - query_c[i];
            sum += diff * diff;
        }
        distance_index[idx].distance = sqrt(sum);
        distance_index[idx].index = idx;
    }
    if (idx >= NUM_RECORDS) {
        distance_index[idx].distance = MAX_DISTANCE;
        distance_index[idx].index = -1;
    }
}

/**
 * Calculate the manhattan distance between query (in constant memory) and every point in the dataset.
 * 
 * @param data Flattened dataset
 * @param distance_index Output vector of distance/index pairs.
*/
__global__ void manhattan_distance_kernel(float *data, distance_index_t *distance_index) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_RECORDS) {
        float sum = 0;
        for (int i = 0; i < NUM_DATA_FEATURES; i++) {
            sum += abs(data[idx * NUM_DATA_FEATURES + i] - query_c[i]);
        }
        distance_index[idx].distance = sum;
        distance_index[idx].index = idx;
    }
    if (idx >= NUM_RECORDS) {
        distance_index[idx].distance = MAX_DISTANCE;
        distance_index[idx].index = -1;
    }
}

/**
 * Calculate the K-nearest-neighbors for h_query, load the result into h_distance_index.
 * 
 * @param h_query Host array representing the query (the record we want the k-neighbors for)
 * @param d_data Flattened, normalized dataset, loaded onto the GPU
 * @param h_distance_index Output vector. Sorted distances and indices for KNeighbors will be loaded back to the host.
*/
void knn(float *h_query, float *d_data, distance_metric_t dist_metric, distance_index_t *h_distance_index) {
    // allocate device memory for the distances and indices

    distance_index_t *d_distance_index;
    cudaMalloc((void **)&d_distance_index, sizeof(distance_index_t) * NUM_BIOTONIC);

    // every thread will access the query features, so store it as constant memory
    cudaMemcpyToSymbol(query_c, h_query, NUM_DATA_FEATURES * sizeof(float)); 

    // each thread handles the distance calculation for 1 record, which is a single row of features
    int num_blocks = (NUM_BIOTONIC + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // launch the appropriate distance kernel
    switch(dist_metric) {
        case EUCLIDEAN:
            printf("Calling Euclidean kernel...\n");
            euclidean_distance_kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, d_distance_index);
            break;
        case MANHATTAN:
            printf("Calling Manhattan kernel...\n");
            manhattan_distance_kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, d_distance_index);
            break;
    }
    
    // get the sorted indices. Size is the number of values being sorted at this iteration
    for (int size = 2; size <= NUM_BIOTONIC; size *= 2) {
        // width is the breath of the comparisons
        for(int width = size; width >= 2; width /= 2) {
            bitonicsort_kernel<<<num_blocks, BLOCK_SIZE>>>(d_distance_index, size, width);
        }
    }

    cudaDeviceSynchronize();

    // copy NUM_NEIGHBORS items back to the distance_index array
    cudaMemcpy(h_distance_index, d_distance_index, sizeof(distance_metric_t) * NUM_NEIGHBORS * 2, cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(d_data);
    cudaFree(d_distance_index);
}

/**
 * Create a new device array and copy the data onto the device.
 * 
 * @param flattened_host_data 1D array for all data, stored on the host
 * @param num_elements The number of elements in flattened_host_data
*/
float *copy_data_to_device(float *flattened_host_data, int num_elements) {
    float *d_data;
    size_t size = num_elements * sizeof(float);
    cudaMalloc((void **)&d_data, size);
    cudaMemcpy(d_data, flattened_host_data, size, cudaMemcpyHostToDevice);
    return d_data;
}
