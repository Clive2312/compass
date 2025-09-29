#include "baseline_utils.h"

#include <vector>
#include <cstdint>
#include <map>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sys/stat.h>  
#include <stdio.h>      
#include <unistd.h>    

#include "evp.h"

// json
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/error/en.h"

// void insecure_load(
//     vector<vector<float>>& doc_scores,
//     vector<vector<int>>& doc_ids,
//     vector<vector<int>>& to_position_map,
//     OramConfig config,
//     RandForOramInterface* rand,
//     vector<int>& position_map
//     ){

//     cout << "Insecure loading..." << endl;

//     size_t per_block_size = (1 + config.block_size);
//     size_t per_bucket_size = config.bucket_size*per_block_size;
//     size_t total_size = config.num_buckets * per_bucket_size;

//     int* bkts = new int[total_size];
//     // Initialize each block id to be -1
//     for(size_t bucket_id = 0; bucket_id < config.num_buckets; bucket_id ++){
//         for(size_t block_id = 0; block_id < config.bucket_size; block_id++){
//             size_t offset = bucket_id*per_bucket_size + block_id*per_block_size;
//             bkts[offset] = -1;
//         }
//     }

//     cout << "Insecure loading 2..." << endl;

//     int keyword_size = doc_scores.size();
//     to_position_map.resize(keyword_size);

//     int cnt = 0;
//     for(int kid = 0; kid < keyword_size; kid++){
//         int doc_size = doc_scores[kid].size();
//         for(int poffset = 0; poffset < doc_size; poffset++){
//             int pid = doc_ids[kid][poffset];
//             // Block b(rss->block_size);
//             // b.index = to_position_map[kid][poffset];
//             // b.data[0] = kid;
//             // b.data[1] = pid;
//             // ((float*)(b.data.data()))[2] = doc_scores[kid][poffset];
//             to_position_map[kid].push_back(cnt);

//             bool written = false;
//             while(!written){
//                 int leaf = rand->getRandomLeafWithBound(config.num_leaves);
//                 vector<Block> blocks;
//                 for (int l = config.num_levels - 1; l >= 0; l--) {
//                     size_t offset = (1<<l) - 1 + (leaf >> (config.num_levels - l - 1));
//                     int* bkt = bkts + offset * per_bucket_size;
//                     for(int block_id = 0; block_id < config.bucket_size; block_id++){
//                         int* block = bkt + block_id * per_block_size;
//                         if(block[0] == -1){
//                             block[0] = cnt;
//                             block[1] = kid;
//                             block[2] = pid;
//                             ((float*)block)[3] = doc_scores[kid][poffset];
//                             // cout << "Assigning "<< cnt << " kid: " << kid << " pid: " << pid << " score: " << doc_scores[kid][poffset] << " to leaf: " << leaf << " bucket:" << block_id << endl;
//                             written = true;
//                             break;
//                         }
//                     }
//                     if(written){
//                         position_map.push_back(leaf);
//                         break;
//                     }
//                 }
//             }
//             cnt++;

//             if(cnt % 1000000 == 0){
//                 cout << "Loading " << cnt << endl;
//             }
//         }
//     }

//     std::vector<SBucket> buckets(config.num_buckets);

//     // encrypt
// 	for(int i = 0; i < config.num_buckets; i++){
// 		unsigned char* payload = (unsigned char*)bkts + i*per_bucket_size;
// 		// Encrypt and write
// 		int ctx_len = encrypt_wrapper(payload, per_bucket_size, buckets[i].data);
// 		assert(ctx_len == SBucket::getCipherSize());

// 		// this->buckets[i]->from_ptr(payload);
// 	}
    
//     cout << " - done!" << endl;

//     delete[] bkts;
//     return;
// }

int parse_json_baseline(string config_path, BaselineMetadata &md, string dataset){
    // Open the JSON file
    ifstream file(config_path);
    if (!file.is_open()) {
        cerr << "Failed to open file." << endl;
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

    // Accessing values
    if (document.HasParseError()) {
        cerr << "Parse error: " << rapidjson::GetParseError_En(document.GetParseError()) << endl;
        return 1;
    }

    // Extracting data
    if (document.HasMember(dataset) && document[dataset].IsObject()) {
        const rapidjson::Value& val = document[dataset];

        if(val.HasMember("block_size") && val["block_size"].IsInt()){
            md.block_size = val["block_size"].GetInt();
        } else{
            cerr << "Parse error: block_size not found or invalid format" << endl;
            return 1;
        } 

        if(val.HasMember("dim") && val["dim"].IsInt()){
            md.dim = val["dim"].GetInt();
        } else{
            cerr << "Parse error: dim not found or invalid format" << endl;
            return 1;
        }

        if(val.HasMember("base_size") && val["base_size"].IsInt()){
            md.base_size = val["base_size"].GetInt();
        } else{
            cerr << "Parse error: base_size not found or invalid format" << endl;
            return 1;
        }

        if(val.HasMember("bucket_size") && val["bucket_size"].IsInt()){
            md.bucket_size = val["bucket_size"].GetInt();
        } else{
            cerr << "Parse error: bucket_size not found or invalid format" << endl;
            return 1;
        }

        if(val.HasMember("keyword_len") && val["keyword_len"].IsInt()){
            md.keyword_len = val["keyword_len"].GetInt();
        } else{
            cerr << "Parse error: keyword_len not found or invalid format" << endl;
            return 1;
        }

        if(val.HasMember("tfidf_score_path") && val["tfidf_score_path"].IsString()){
            md.tfidf_score_path = val["tfidf_score_path"].GetString();
        } else{
            cerr << "Parse error: tfidf_score_path not found or invalid format" << endl;
            return 1;
        } 

        if(val.HasMember("tfidf_query_path") && val["tfidf_query_path"].IsString()){
            md.tfidf_query_path = val["tfidf_query_path"].GetString();
        } else{
            cerr << "Parse error: tfidf_query_path not found or invalid format" << endl;
            return 1;
        } 

        if(val.HasMember("buckets_path") && val["buckets_path"].IsString()){
            md.buckets_path = val["buckets_path"].GetString();
        } else{
            cerr << "Parse error: buckets_path not found or invalid format" << endl;
            return 1;
        } 


        if(val.HasMember("pos_map_path") && val["pos_map_path"].IsString()){
            md.pos_map_path = val["pos_map_path"].GetString();
        } else{
            cerr << "Parse error: pos_map_path not found or invalid format" << endl;
            return 1;
        } 
    }

    return 0;

}

void read_score(const char* fname, vector<vector<float>>& scores, vector<vector<int>>& ids, size_t& ntotal, size_t& max_docs, int keyword_len) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }

    scores.resize(keyword_len);
    ids.resize(keyword_len);

    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % (3 * 4) == 0 || !"weird file size");
    ntotal = sz / (3 * 4);

    for (size_t i = 0; i < ntotal; i++){
        int pid;
        int kid;
        float score;

        fread(&pid, 1, sizeof(int), f);
        fread(&kid, 1, sizeof(int), f);
        fread(&score, 1, sizeof(float), f);

        // if(scores.find(kid) == scores.end()){
        //     scores[kid] = map<int, float>();
        // } 

        scores[kid].push_back(score);
        ids[kid].push_back(pid);
    }

    max_docs = 0;

    // Get stats
    for(auto& it : scores){
        max_docs = max(max_docs, it.size());
    }

    cout << "max doc size: " << max_docs << endl; 
    cout << "ntotal: " << ntotal << endl; 
}