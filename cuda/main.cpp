#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include "support.h"
#include "knn.h"

/**
 * Get an array of size NUM_SONGS * NUM_SONG_FEATURES filled with song data.
 *  To access song i's features, use: data[i * NUM_SONG_FEATURES + j]
 *  where 0 <= j < NUM_SONG_FEATURES.
 * 
 * @param filename filename of CSV file containing normalized song feature data
*/
float *readCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return nullptr;
    }

    // allocate one big array for the song data
    float *data = new float[NUM_SONGS * NUM_SONG_FEATURES];
    if (data == nullptr) {
        std::cerr << "Memory allocation failed." << std::endl;
        return nullptr;
    }

    std::string line;
    int index = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            data[index++] = std::stof(value);
        }
    }

    file.close();
    return data;
}

/**
 * Return a newly allocated float[NUM_SONG_FEATURES], filled with features of the song at "index".
 * 
 * @param index Index of the song in the dataset. ex: if index == 5, return the array of song 5's features.
 * @param data 1D array storing song data.
*/
float *get_query_array(int index, float *data) {
    float *query = new float[NUM_SONG_FEATURES];
    for (int i = 0; i < NUM_SONG_FEATURES; i++) {
        query[i] = data[index * NUM_SONG_FEATURES + i];
    }
    return query;
}

int main() {
    // read the data into a big 1D array
    std::string filename = "normalized_knn_data.csv";
    float *data = readCSV(filename);

    if (data == nullptr) {
        return 1;
    }

    int query_index = 9;
    float *query = get_query_array(query_index, data);
    float *h_distances = new float[NUM_NEIGHBORS];
    float *h_indices = new float[NUM_NEIGHBORS];
    float *d_data = copy_data_to_device(data, NUM_SONGS * NUM_SONG_FEATURES);

    // print the first 10 rows of the matrix
    printf("\nFirst 10 rows of data matrix:\n");
    for (int i = 0; i < 10; ++i) {
        printf("song %d: ", i);
        for (int j = 0; j < NUM_SONG_FEATURES; ++j) {
            printf("%.6f", data[i * NUM_SONG_FEATURES + j]);
            if (j < NUM_SONG_FEATURES - 1) {
                printf(", ");
            }
        }
        printf("\n");
    }

    // let's make sure query is correct
    printf("\nquery : ", query_index);
    for (int i = 0; i < NUM_SONG_FEATURES; i++) {
        printf("%.6f, ", query[i]);
    }
    printf("\n\n");

    // call the knn algorithm to launch the GPU kernels
    knn(query, d_data, EUCLIDEAN, h_distances, h_indices);

    // cleanup
    delete[] data;
    delete[] h_distances;
    delete[] h_indices;
    delete[] query;
}
