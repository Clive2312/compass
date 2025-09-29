#include "cluster_config_parser.h"

#include <iostream>
#include <cassert>
#include <cstdio>
#include <fstream>

// json
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/error/en.h"

int parseClusterJson(const string& config_path, ClusterMetadata& md, const string& dataset) {
    // Open the JSON file
    ifstream file(config_path);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << config_path << endl;
        return 1;
    }

    // Read the entire file into a buffer
    string jsonContent;
    string line;
    while (getline(file, line)) {
        jsonContent += line + "\n";
    }
    file.close();

    // Parse the JSON content
    rapidjson::Document document;
    document.Parse(jsonContent.c_str());

    // Check for parse errors
    if (document.HasParseError()) {
        cerr << "Parse error: " << rapidjson::GetParseError_En(document.GetParseError()) << endl;
        return 1;
    }

    // Set default values
    md.slot_count = 8192;
    md.num_queries = 1;
    md.k = 10;

    // Extract dataset-specific configuration
    if (document.HasMember(dataset.c_str()) && document[dataset.c_str()].IsObject()) {
        const rapidjson::Value& val = document[dataset.c_str()];

        // Parse paths
        if (val.HasMember("base_path") && val["base_path"].IsString()) {
            md.base_path = val["base_path"].GetString();
        } else {
            cerr << "Parse error: base_path not found or invalid format for dataset: " << dataset << endl;
            return 1;
        }

        if (val.HasMember("query_path") && val["query_path"].IsString()) {
            md.query_path = val["query_path"].GetString();
        } else {
            cerr << "Parse error: query_path not found or invalid format for dataset: " << dataset << endl;
            return 1;
        }

        if (val.HasMember("gt_path") && val["gt_path"].IsString()) {
            md.gt_path = val["gt_path"].GetString();
        } else {
            cerr << "Parse error: gt_path not found or invalid format for dataset: " << dataset << endl;
            return 1;
        }

        if (val.HasMember("centroid_path") && val["centroid_path"].IsString()) {
            md.centroid_path = val["centroid_path"].GetString();
        } else {
            cerr << "Parse error: centroid_path not found or invalid format for dataset: " << dataset << endl;
            return 1;
        }

        if (val.HasMember("result_path") && val["result_path"].IsString()) {
            md.result_path = val["result_path"].GetString();
        } else {
            cerr << "Parse error: result_path not found or invalid format for dataset: " << dataset << endl;
            return 1;
        }

        if (val.HasMember("assignment_path") && val["assignment_path"].IsString()) {
            md.assignment_path = val["assignment_path"].GetString();
        } else {
            cerr << "Parse error: assignment_path not found or invalid format for dataset: " << dataset << endl;
            return 1;
        }

        // Parse numerical values
        // Config values for kmeans_cluster
        if (val.HasMember("config_n_centroids") && val["config_n_centroids"].IsInt()) {
            md.config_nc = val["config_n_centroids"].GetInt();
        } else {
            cerr << "Parse error: config_n_centroids not found or invalid format for dataset: " << dataset << endl;
            return 1;
        }

        if (val.HasMember("config_max_nodes_per_centroids") && val["config_max_nodes_per_centroids"].IsInt()) {
            md.config_max_node_per_cluster = val["config_max_nodes_per_centroids"].GetInt();
        } else {
            cerr << "Parse error: config_max_nodes_per_centroids not found or invalid format for dataset: " << dataset << endl;
            return 1;
        }

        // Actual values for cluster_search
        if (val.HasMember("nc") && val["nc"].IsInt()) {
            md.nc = val["nc"].GetInt();
        } else {
            cerr << "Parse error: nc not found or invalid format for dataset: " << dataset << endl;
            return 1;
        }

        if (val.HasMember("node_per_cluster") && val["node_per_cluster"].IsInt()) {
            md.node_per_cluster = val["node_per_cluster"].GetInt();
        } else {
            cerr << "Parse error: node_per_cluster not found or invalid format for dataset: " << dataset << endl;
            return 1;
        }

        // Set dimension based on dataset
        if (dataset == "laion") {
            md.dim = 512;
        } else if (dataset == "sift") {
            md.dim = 128;
        } else if (dataset == "trip") {
            md.dim = 768;
        } else if (dataset == "msmarco") {
            md.dim = 768;
        } else {
            cerr << "Unknown dataset: " << dataset << endl;
            return 1;
        }

    } else {
        cerr << "Dataset '" << dataset << "' not found in config file" << endl;
        return 1;
    }

    return 0;
}