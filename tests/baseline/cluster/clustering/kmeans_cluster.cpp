
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <map>
#include <iostream>
#include <string>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/time.h>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_factory.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/index_io.h>

// Include cluster config parser and ArgMapping
#include "../cluster_config_parser.h"
#include "ArgMapping.h"

using namespace std;

float compute_recall(faiss::idx_t* gt, int gt_size, faiss::idx_t* I, int nq, int k, int gamma=1) {
    // printf("compute_recall params: gt.size(): %ld, gt_size: %d, I.size(): %ld, nq: %d, k: %d, gamma: %d\n", gt.size(), gt_size, I.size(), nq, k, gamma);
    int n_1 = 0, n_10 = 0, n_100 = 0;
    for (int i = 0; i < nq; i++) { // loop over all queries
        // int gt_nn = gt[i * k];
        faiss::idx_t* first = gt + i*gt_size;
        faiss::idx_t* last = gt + i*gt_size + (k / gamma);
        std::vector<faiss::idx_t> gt_nns_tmp(first, last);
        // if (gt_nns_tmp.size() > 10) {
        //     printf("gt_nns size: %ld\n", gt_nns_tmp.size());
        // }
        // gt_nns_tmp.resize(k); // truncate if gt_size > k
        // std::set<faiss::idx_t> gt_nns_100(gt_nns_tmp.begin(), gt_nns_tmp.end());
        // gt_nns_tmp.resize(10);
        std::set<faiss::idx_t> gt_nns_10(gt_nns_tmp.begin(), gt_nns_tmp.end());
        gt_nns_tmp.resize(1);
        std::set<faiss::idx_t> gt_nns_1(gt_nns_tmp.begin(), gt_nns_tmp.end());
        // if (gt_nns.size() > 10) {
        //     printf("gt_nns size: %ld\n", gt_nns.size());
        // }
        for (int j = 0; j < k; j++) { // iterate over returned nn results
            // if (gt_nns_100.count(I[i * k + j])!=0) {
            //     if (j < 100 * gamma)
            //         n_100++;
            // }
            if (gt_nns_10.count(I[i * k + j])!=0) {
                if (j < 10 * gamma)
                    n_10++;
            }
            if (gt_nns_1.count(I[i * k + j])!=0) {
                if (j < 1 * gamma)
                    n_1++;
            }
        }
    }
    // BASE ACCURACY
    printf("* Base HNSW accuracy relative to exact search:\n");
    printf("\tR@1 = %.4f\n", n_1 / float(nq) );
    printf("\tR@10 = %.4f\n", n_10 / float(nq));
    // printf("\tR@100 = %.4f\n", n_100 / float(nq)); // not sure why this is always same as R@10
    // printf("\t---Results for %ld queries, k=%d, N=%ld, gt_size=%d\n", nq, k, N, gt_size);
    return (n_10 / float(nq));
}

std::vector<std::vector<float>> tiptoe_clustering_recursive(
        size_t dim,
        size_t nb,
        size_t n_centroids,
        const float* xb,
        size_t max_n_nodes) {

    std::cout << "-- tiptoe_clustering_recursive on: " << nb << " nodes to: " << n_centroids << " centroids" << std::endl;

    std::vector<std::vector<float>> centroids;

    // Compute centroids
    faiss::Clustering clus(dim, n_centroids);
    clus.nredo = 1;
    clus.niter = 25;
    clus.spherical = true;
    clus.verbose = dim * nb * n_centroids > (size_t(1) << 30);
    // display logs if > 1Gflop per iteration
    faiss::IndexFlatIP index(dim);
    clus.train(nb, xb, index); 
    // memcpy(centroids, clus.centroids.data(), sizeof(*centroids) * dim * n_centroids);

    // Get assignment
    std::map<size_t, std::vector<size_t>> assignment;
    faiss::IndexFlatIP index_centroids(dim);
    index_centroids.add(n_centroids, clus.centroids.data());

    faiss::idx_t* i_b = new faiss::idx_t[nb];
    float* d_b = new float[nb];

    index_centroids.search(nb, xb, 1, d_b, i_b);

    for(size_t i = 0; i < nb; i++){
        assignment[i_b[i]].push_back(i);
    }

    // check size
    for(size_t nc = 0; nc < n_centroids; nc++){
        if(assignment[nc].size() > max_n_nodes){
            int new_n_centrods = ceil(assignment[nc].size() * 1.0 / max_n_nodes); 

            if(new_n_centrods < n_centroids){
                float* new_xb = new float[dim*assignment[nc].size()];

                for(size_t i = 0; i < assignment[nc].size(); i++){
                    size_t ni = assignment[nc][i];
                    memcpy(new_xb + i * dim, xb + ni * dim, dim*sizeof(float));
                }
                std::cout << " ## RECURSIVE." << std::endl;

                std::vector<std::vector<float>> new_centroids = tiptoe_clustering_recursive(
                    dim, 
                    assignment[nc].size(),
                    new_n_centrods,
                    new_xb,
                    max_n_nodes
                );

                for(auto c : new_centroids){
                    centroids.push_back(c);
                }

                std::cout << " ## RECURSIVE DONE." << std::endl;
            } else{
                std::vector<float> c(
                    clus.centroids.data() + nc*dim,
                    clus.centroids.data() + (nc + 1)*dim
                );
                centroids.push_back(c);
            }
        } else{
            std::vector<float> c(
                clus.centroids.data() + nc*dim,
                clus.centroids.data() + (nc + 1)*dim
            );
            centroids.push_back(c);
        }
    }
    
    return centroids;
}

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

void fvecs_write(const char* fname, float* data, size_t d, size_t n) {
    FILE* f = fopen(fname, "w");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    for (size_t i = 0; i < n; i++){
        fwrite(&d, 1, sizeof(int), f);
        fwrite(data + i*d, d, sizeof(float), f);
    }
    fclose(f);
}

void ivecs_write(const char* fname, int* data, size_t d, size_t n) {
    fvecs_write(fname, (float*)data, d, n);
}

string dataset = "";
bool rebuild = true;

int main(int argc, char **argv) {
    ArgMapping amap;
    amap.arg("d", dataset, "Dataset: [sift, trip, laion]");
    amap.arg("rebuild", rebuild, "Rebuild centroids (default: true, set false to use existing)");
    amap.parse(argc, argv);

    if (dataset.empty()) {
        cerr << "Error: Dataset must be specified with -d option" << endl;
        cerr << "Available datasets: sift, trip, laion" << endl;
        return 1;
    }

    // Parse configuration from JSON
    ClusterMetadata md;
    cout << "-> Parsing json..." << endl;
    if(parseClusterJson("../tests/baseline/cluster/clustering/config.json", md, dataset) != 0) {
        cerr << "Early exit..." << endl;
        return 0;
    }

    size_t nb;
    size_t d;

    // Extract configuration values (using config values for clustering)
    size_t n_centroids = md.config_nc;
    size_t max_nodes_per_centroids = md.config_max_node_per_cluster;
    string base_path = md.base_path;
    string query_path = md.query_path;
    string gt_path = md.gt_path;
    string centroid_path = md.centroid_path;
    string result_path = md.result_path;
    string assignment_path = md.assignment_path;

    cout << "Configuration for " << dataset << ":" << endl;
    cout << "  - n_centroids: " << n_centroids << endl;
    cout << "  - max_nodes_per_centroids: " << max_nodes_per_centroids << endl;
    cout << "  - base_path: " << base_path << endl;
    cout << "  - query_path: " << query_path << endl;
    cout << "  - gt_path: " << gt_path << endl;
    cout << "  - centroid_path: " << centroid_path << endl;
    cout << "  - result_path: " << result_path << endl;
    cout << "  - assignment_path: " << assignment_path << endl;

    cout << "Loading base data from: " << base_path << endl;
    float* xb = fvecs_read(base_path.c_str(), &d, &nb);

    // Loading Queries
    std::cout << "Loading queries..." << std::endl;
    size_t nq;
    float* xq;
    size_t d2;
    int n_result = 10;

    cout << "Loading queries from: " << query_path << endl;
    xq = fvecs_read(query_path.c_str(), &d2, &nq);

    std::cout << "Clustering data: " << std::endl;
    std::cout << "- d: " << d << std::endl;
    std::cout << "- n: " << nb << std::endl;
    std::cout << "- nc: " << n_centroids << std::endl;

    faiss::Index* index = nullptr;

    if (!rebuild) {
        // Load existing index from file
        std::cout << "Loading existing centroids from: " << centroid_path << std::endl;
        index = faiss::read_index(centroid_path.c_str());
        n_centroids = index->ntotal;
        std::cout << "Loaded index with " << n_centroids << " centroids" << std::endl;
    } else {
        // Perform clustering to find centroids
        // tiptoe_clustering(
        //     d,
        //     nb,
        //     n_centroids,
        //     xb,
        //     centroids
        // );

        std::vector<std::vector<float>> cs = tiptoe_clustering_recursive(
            d,
            nb,
            n_centroids,
            xb,
            max_nodes_per_centroids
        );

        n_centroids = cs.size();
        float* centroids = new float[d*n_centroids];

        for(int i = 0; i < n_centroids; i++){
            memcpy(centroids + i*d, cs[i].data(), d*sizeof(float));
        }
        std::cout << "New n_centrods: " << n_centroids << std::endl;

        std::cout << "Building flat index..." << std::endl;

        const char* index_key = "Flat";
        index = faiss::index_factory(d, index_key);
        index->add(n_centroids, centroids);

        cout << "Saving index to: " << centroid_path << endl;
        faiss::write_index(index, centroid_path.c_str());

        delete[] centroids;
    }

    int n_dup_cluster = 2;
    faiss::idx_t* i_b = new faiss::idx_t[nb * n_dup_cluster];
    float* d_b = new float[nb * n_dup_cluster];


    std::cout << "Searching on flat index..." << std::endl;
    index->search(nb, xb, n_dup_cluster, d_b, i_b);

    std::cout << "Sorting diff..." << std::endl;
    std::vector<float> D_diff;

    for(int i = 0; i < nb; i++){
        D_diff.push_back(d_b[2*i + 1] - d_b[2*i]);
    }

    for(int i = 0; i < 10; i++){
        std::cout << D_diff[i] << " "; 
    }
    std::cout << std::endl;

    std::sort(D_diff.begin(), D_diff.end(), [](float a, float b) {
        return a < b;
    });

    for(int i = 0; i < 10; i++){
        std::cout << D_diff[i] << " "; 
    }
    std::cout << std::endl;

    std::cout << "Sorting diff..." << std::endl;
    int cutoff = 0.2*nb;
    float cutoff_d = D_diff[cutoff];
    std::cout << "Cut off diff: " << cutoff_d << std::endl;

    std::map<int, std::vector<int>> assignment;

    int cnt = 0;
    for(int i = 0; i < nb; i++){
        int centroid_1 = i_b[2*i];
        int centroid_2 = i_b[2*i + 1];
        assignment[centroid_1].push_back(i);
        if(D_diff[i] < cutoff_d){
            assignment[centroid_2].push_back(i);
            cnt ++;
        }
    }

    // Save assignment to file
    std::cout << "Saving assignment to: " << assignment_path << std::endl;
    FILE* f_assign = fopen(assignment_path.c_str(), "wb");
    if (!f_assign) {
        std::cerr << "Error: Could not open assignment file for writing: " << assignment_path << std::endl;
    } else {
        // Write total number of clusters
        int num_clusters = assignment.size();
        fwrite(&num_clusters, sizeof(int), 1, f_assign);

        // Write each cluster's data (variable size per cluster)
        for (const auto& cluster : assignment) {
            int cluster_id = cluster.first;
            int cluster_size = cluster.second.size();  // This cluster's specific size

            // Write cluster ID and its size
            fwrite(&cluster_id, sizeof(int), 1, f_assign);
            fwrite(&cluster_size, sizeof(int), 1, f_assign);

            // Write all node IDs for this cluster (variable length)
            fwrite(cluster.second.data(), sizeof(int), cluster_size, f_assign);
        }
        fclose(f_assign);
        std::cout << "Assignment saved successfully with " << num_clusters << " clusters" << std::endl;
    }

    // Break large clusters
    // int max_cnt_2 = 0;
    // for(int cid = 0; cid < n_centroids; cid++){
    //     int cluster_size = assignment[cid].size();
    //     // std::cout << "Cluster contains "<< cluster_size << " nodes." << std::endl;
    //     max_cnt_2 = cluster_size > max_cnt_2 ? cluster_size : max_cnt_2;
    // }

    // std::cout << "Largest cluster contains "<< max_cnt_2 << " nodes." << std::endl;

    // return 0;

    // Constructing Index for each clusters
    std::map<std::pair<int, int>, int> reverse_map;
    std::cout << "Building index for each cluster..." << std::endl;
    std::vector<faiss::Index*> clusters;
    faiss::index_factory_verbose = 0;
    const char* flat_key = "Flat";
    int max_cnt = 0;
    for(int cid = 0; cid < n_centroids; cid++){
        faiss::Index* index = faiss::index_factory(d, flat_key);
        if(assignment.find(cid) == assignment.end()){
            assert(0);
        }
        int cnt = 0;
        for(auto nid : assignment[cid]){
            index->add(1, xb + nid*d);
            reverse_map[std::make_pair(cid, cnt)] = nid;
            cnt++;
        }
        // std::cout << cnt << " nodes added to cluster "<< cid << std::endl;
        max_cnt = cnt > max_cnt ? cnt : max_cnt;
        clusters.push_back(index);
    }

    std::cout << "Total number of clusters "<< n_centroids  << std::endl;
    std::cout << "Largest cluster contains "<< max_cnt << " nodes." << std::endl;

    faiss::idx_t* I = new faiss::idx_t[nq*n_result];
    float* D = new float[nq*n_result];
    
    faiss::idx_t* i_q = new faiss::idx_t[nq];
    float* d_q = new float[nq];
    index->search(nq, xq, 1, d_q, i_q);

    std::cout << "Searching..." << std::endl;
    for(int qid = 0; qid < nq; qid++){
        int cid = i_q[qid];
        // std::cout << qid << " query to cluster " << cid << std::endl;
        clusters[cid]->search(
            1, 
            xq + qid*d, 
            n_result, 
            D + qid*n_result, 
            I + qid*n_result
        );
    }

    for(int qid = 0; qid < nq; qid++){
        int cid = i_q[qid];
        for(int rid = 0; rid< n_result; rid++){
            I[qid*n_result + rid] = reverse_map[std::make_pair(cid, I[qid*n_result + rid])];
        }
    }

    size_t k;         // nb of results per query in the GT
    faiss::idx_t* gt; // nq * k matrix of ground-truth nearest-neighbors

    std::cout << "Loading GT..." << std::endl;
    {

        // load ground-truth and convert int to long
        size_t nq2;
        cout << "Loading ground truth from: " << gt_path << endl;
        int* gt_int = ivecs_read(gt_path.c_str(), &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        gt = new faiss::idx_t[k * nq];
        for (int i = 0; i < k * nq; i++) {
            gt[i] = gt_int[i];
        }
        delete[] gt_int;
    }

    {
        // We only need k = 10
        int k0 = 10;
        faiss::idx_t* gt0 = new faiss::idx_t[k0 * nq];
        for(int i = 0; i < nq; i++){
            for(int j = 0; j < k0; j++){
                gt0[i*k0 + j] = gt[i*k + j];
            }
        }

        k = k0;
        gt = gt0;
    }

    std::cout << "Computing Recall..." << std::endl;
    compute_recall(
        gt,
        k,
        I,
        nq,
        k
    );

    int* result = new int[nq * k];
    for(int i = 0; i < nq*k; i++){
        result[i] = I[i];
    }

    cout << "Saving results to: " << result_path << endl;
    ivecs_write(result_path.c_str(), result, k, nq);

    return 0;
}