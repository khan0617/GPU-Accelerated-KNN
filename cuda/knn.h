#ifndef KNN_H
#define KNN_H

float *copy_data_to_device(float *flattened_host_data, int num_elements);
void knn(float *h_query, float *d_data, distance_metric_t dist_metric, distance_index_t *h_distance_index);

#endif // KNN_H

