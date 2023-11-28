#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>

#define NUM_SONGS 170653
#define NUM_SONG_FEATURES 9

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
    float* data = (float *)malloc(NUM_SONGS * NUM_SONG_FEATURES * sizeof(float));
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

int main() {
    std::string filename = "normalized_knn_data.csv";
    float* data = readCSV(filename);

    if (data == nullptr) {
        return 1;
    }

    // print the first 10 rows of the matrix
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < NUM_SONG_FEATURES; ++j) {
            printf("%.6f", data[i * NUM_SONG_FEATURES + j]);
            if (j < NUM_SONG_FEATURES - 1) {
                printf(", ");
            }
        }
        printf("\n");
    }

    // data cleanup
    free(data);
}
