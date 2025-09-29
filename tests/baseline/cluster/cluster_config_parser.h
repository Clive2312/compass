#ifndef CLUSTER_CONFIG_PARSER_H
#define CLUSTER_CONFIG_PARSER_H

#include <string>

using namespace std;

struct ClusterMetadata {
    // Dataset configuration
    int nc;                      // number of centroids/clusters
    int dim;                     // dimension
    int node_per_cluster;        // max nodes per cluster

    int config_nc;
    int config_max_node_per_cluster;

    // HE parameters
    size_t slot_count;           // slot count for BFV
    int num_queries;            // number of queries
    int k;                      // top-k results

    // File paths
    string base_path;
    string query_path;
    string gt_path;
    string centroid_path;
    string result_path;
    string assignment_path;
};

int parseClusterJson(const string& config_path, ClusterMetadata& md, const string& dataset);

#endif // CLUSTER_CONFIG_PARSER_H