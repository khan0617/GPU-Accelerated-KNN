// definitions.h
#ifndef SUPPORT_H
#define SUPPORT_H

#define NUM_SONGS 170653
#define NUM_BIOTONIC 262144 //Smallest power of 2 that holds NUM_SONGS
#define NUM_SONG_FEATURES 9
#define NUM_NEIGHBORS 10 // how many nearest neighbors to get
#define MAX_DISTANCE 20

typedef enum {
    EUCLIDEAN = 0,
    MANHATTAN
} distance_metric_t;

typedef struct {
    float distance;
    int index;
} distance_index_t;

#endif // SUPPORT_H
