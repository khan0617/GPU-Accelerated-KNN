#include <stdio.h>
#include "support.h"

#define BLOCK_SIZE 256
__constant__ float query_c[NUM_SONG_FEATURES];


/**
 * Return the indices that would sort the distances array.
 * 
 * @param distances Device array of knn distances across the whole dataset
 * @param indices Output vector
*/
__global__ void argsort_kernel(float *distances, float *indices) {
    // TODO
    return;
}

/**
 * Calculate the euclidean distance between query (in constant memory) and every point in the dataset.
 * 
 * @param data Flattened dataset with NUM_SONGS * NUM_SONG_FEATURES elements.
 * @param distances The output vector
 *  ex: distances[0] is the distance between data[0] and query point.
*/
__global__ void euclidean_distance_kernel(float *data, float *distances) {
    // TODO
    return;
}

/**
 * Calculate the manhattan distance between query (in constant memory) and every point in the dataset.
 * 
 * @param data Flattened dataset
 * @param distances The output vector
*/
__global__ void manhattan_distance_kernel(float *data, float *distances) {
    // TODO
    return;
}

/**
 * Launch the appropriate kernels and return distances, indices.
 * 
 * @param h_query Host array representing the query (the record we want the k-neighbors for)
 * @param d_data Flattened, normalized song data, loaded onto the GPU
 * @param h_distances The k-nearest-neighbor distances will be loaded here after computation
 * @param h_indices The indices of the k-nearest-neighbors will be loaded here
*/
void knn(float *h_query, float *d_data, distance_metric_t dist_metric, float *h_distances, float *h_indices) {
    printf("hey from knn host function!\n");

    // allocate device memory for the distances and indices
    float *d_distances, *d_indices;
    cudaMalloc((void **)&d_distances, sizeof(float) * NUM_SONGS);
    cudaMalloc((void **)&d_indices, sizeof(float) * NUM_SONGS);

    // every thread will acces the query features, so store it as constant memory
    cudaMemcpyToSymbol(query_c, h_query, NUM_SONG_FEATURES * sizeof(float)); 

    // each thread handles the distance calculation for 1 song, which is a single row of featuers
    int num_blocks = (NUM_SONGS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // launch the appropriate distance kernel
    switch(dist_metric) {
        case EUCLIDEAN:
            euclidean_distance_kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, d_distances);
            break;
        case MANHATTAN:
            manhattan_distance_kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, d_distances);
            break;
    }
    
    // get the sorted indices
    argsort_kernel<<<num_blocks, BLOCK_SIZE>>>(d_distances, d_indices);
    cudaDeviceSynchronize();

    // copy NUM_NEIGHBORS items back to h_distances and h_indices
    cudaMemcpy(h_distances, d_distances, sizeof(float) * NUM_NEIGHBORS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_indices, d_indices, sizeof(float) * NUM_NEIGHBORS, cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(d_distances);
    cudaFree(d_indices);
    cudaFree(d_data);
}

/**
 * Create a new device array and copy the song data onto the device.
 * 
 * @param flattened_host_data 1D array for all song data, stored on the host
 * @param num_elements The number of elements in flattened_host_data
*/
float *copy_data_to_device(float *flattened_host_data, int num_elements) {
    float *d_data;
    size_t size = num_elements * sizeof(float);
    cudaMalloc((void **)&d_data, size);
    cudaMemcpy(d_data, flattened_host_data, size, cudaMemcpyHostToDevice);
    return d_data;
}
