#include <string>
#include <vector>

using namespace std;

struct BaselineMetadata {
    int dim;
    int base_size;
    int block_size;
    int bucket_size;
    int keyword_len;

    string tfidf_score_path;
    string tfidf_query_path;
    string buckets_path;
    string pos_map_path;
};

int parse_json_baseline(string config_path, BaselineMetadata &md, string dataset);

void read_score(const char* fname, vector<vector<float>>& scores, vector<vector<int>>& ids, size_t& ntotal, size_t& max_docs, int keyword_len);