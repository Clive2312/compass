
// stl
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include <sys/stat.h>
#include <sys/time.h>
#include <numeric>  

#include "ArgMapping.h"
#include "baseline_utils.h"

using namespace std;

int k = 10;

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

// not very clean, but works as long as sizeof(int) == sizeof(float)
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

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

inline void tprint(string s, double t0){
    cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t0 << "s] " << s;
}

int truncate_size = 10000;
string dataset = "";
string f_accuracy = "";

int main(int argc, char **argv){

    ArgMapping amap;
    amap.arg("d", dataset, "Dataset: [trip, msmarco]");
    amap.arg("trunc", truncate_size, "truncation of docs");
    amap.arg("f_accuracy", f_accuracy, "Save accuracy");
    amap.parse(argc, argv);

    BaselineMetadata md;

    cout << "-> Parsing json..." << endl;
    if(parse_json_baseline("../tests/baseline/baseline_config.json", md, dataset) != 0) {
        cerr << "Early exit..." << endl;
        return 0;
    }

    size_t ntotal;
    size_t max_docs;
    vector<vector<float>> doc_scores;
    vector<vector<int>> doc_ids;

    cout << "-> Reading score..." << endl;
    read_score(md.tfidf_score_path.c_str(), doc_scores, doc_ids, ntotal, max_docs, md.keyword_len);

    double t0 = elapsed();

    int* xq;
    size_t nq;
    size_t dq;
    {
        tprint("Loading queries... \n", t0);
        xq = ivecs_read(md.tfidf_query_path.c_str(), &dq, &nq);
        cout << "Loaded " << nq << "queries padded with " << dq << " words" << endl; 
    }

    int* res = new int[nq*k];

    #pragma omp parallel for
    for(int i = 0; i < nq; i++){
        map<int, float> match_docs;
        for(int j = 0; j < dq; j++){
            int key_id = xq[i*dq + j];

            if(key_id >= 0){
                vector<int> docs = doc_ids[key_id];
                vector<float> scores = doc_scores[key_id];

                std::vector<int> indices(docs.size());
                std::iota(indices.begin(), indices.end(), 0);

                std::sort(indices.begin(), indices.end(), [&scores](int i1, int i2) {
                    return scores[i1] > scores[i2];
                });

                vector<int> sorted_docs(docs.size());
                vector<float> sorted_scores(scores.size());

                for (size_t i = 0; i < indices.size(); ++i) {
                    sorted_docs[i] = docs[indices[i]];
                    sorted_scores[i] = scores[indices[i]];
                }

                int real_l = truncate_size;
                if(sorted_docs.size() < real_l){
                    real_l = sorted_docs.size();
                }

                for(int p = 0; p < real_l; p++){
                    int doc_id = sorted_docs[p];
                    float score = sorted_scores[p];

                    if(match_docs.find(doc_id) != match_docs.end()){
                        match_docs[doc_id] += score;
                    } else{
                        match_docs[doc_id] = score;
                    }
                }
            }
        }

        vector<pair<int, double>> vec(match_docs.begin(), match_docs.end());

        std::sort(vec.begin(), vec.end(), 
        [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
            return a.second > b.second;
        });

        int cnt = 0;
        for (const auto& [key, value] : vec) {
            if(cnt >= k){
                break;
            }
            res[i*k + cnt] = key;
            cnt ++;
        }
    }

    if(f_accuracy != ""){
        // write result if search over the whole query set
        // string result_path = dataset + ".ivecs";
        ivecs_write(
            f_accuracy.c_str(), 
            res, 
            k,
            nq
        );
    }

    return 0;
}