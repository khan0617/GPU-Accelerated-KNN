// definitions.h
#ifndef SUPPORT_H
#define SUPPORT_H

#define NUM_RECORDS 170653  // how many points in the dataset (likely the number of rows in a csv.)
#define NUM_BIOTONIC 262144 // Smallest power of 2 that holds NUM_RECORDS
#define NUM_DATA_FEATURES 9 // how many features each point in dataset has (number of columns in a csv.)
#define NUM_NEIGHBORS 10    // how many nearest neighbors to get
#define MAX_DISTANCE 20
#define DATASET_FILENAME "normalized_knn_data.csv"

typedef enum {
    EUCLIDEAN = 0,
    MANHATTAN
} distance_metric_t;

typedef struct {
    float distance;
    int index;
} distance_index_t;

#endif // SUPPORT_H
