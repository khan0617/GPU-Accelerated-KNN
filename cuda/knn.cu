

__constant__ float query[NUM_SONG_FEATURES];


/**
 * Calculate the euclidean distance between query (in constant memory) and every point in the dataset.
 * 
 * @param data Flattened dataset with NUM_SONGS * NUM_SONG_FEATURES elements.
 * @param distances The output vector
 *  ex: distances[0] is the distance between data[0] and query point.
*/
__device__ void euclidean_distance_kernel(float *data, float *distances) {
    return;
}

/**
 * Calculate the manhattan distance between query (in constant memory) and every point in the dataset.
 * 
 * @param data Flattened dataset
 * @param distances The output vector
*/
__device__ void manhattan_distance_kernel(float *data, float *distances) {
    return;
}