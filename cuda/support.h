// definitions.h
#ifndef SUPPORT_H
#define SUPPORT_H

#define NUM_SONGS 170653
#define NUM_SONG_FEATURES 9
#define NUM_NEIGHBORS 10 // how many nearest neighbors to get

typedef enum {
    EUCLIDEAN = 0,
    MANHATTAN
} distance_metric_t;

#endif // SUPPORT_H
