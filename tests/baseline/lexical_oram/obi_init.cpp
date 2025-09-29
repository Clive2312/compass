#include "ArgMapping.h"
#include "net_io_channel.h"

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

// oram
// #include "RemoteServerStorage.h"
// #include "RemoteServer.h"
// # include "RemotePath.h"
#include "RandForOramInterface.h"
#include "RandomForOram.h"

// graph
#include "baseline_utils.h"
#include "../../src/faiss/impl/HNSW.h"
#include "../../src/faiss/impl/ProductQuantizer.h"
#include "../../src/faiss/utils/io.h"
#include "../../src/faiss/index_io.h"

int k = 10;

using namespace std;

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

inline void tprint(string s, double t0){
    cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t0 << "s] " << s;
}


void save_position_map(const char* fname, std::vector<int>& pos_map){
    int s = pos_map.size();
    FILE* f = fopen(fname, "w");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    fwrite(&s, 1, sizeof(int), f);
    fwrite(pos_map.data(), 1, pos_map.size()*sizeof(int), f);
    fclose(f);
}

string dataset = "";

int main(int argc, char **argv) {
    ArgMapping amap;
    amap.arg("d", dataset, "Dataset: [trip, msmarco]");
    amap.parse(argc, argv);

    BaselineMetadata md;

    cout << "-> Parsing json..." << endl;
    if(parse_json_baseline("../tests/baseline/baseline_config.json", md, dataset) != 0) {
        cerr << "Early exit..." << endl;
        return 0;
    }

    double t0 = elapsed();

    size_t ntotal;
    size_t max_docs;
    vector<vector<float>> doc_scores;
    vector<vector<int>> doc_ids;
    vector<vector<int>> doc_to_position_map;
    vector<int> position_map;

    tprint("Start reading... \n", t0);

    read_score(md.tfidf_score_path.c_str(), doc_scores, doc_ids, ntotal, max_docs, md.keyword_len);

    cout << "ntotal: " << ntotal << endl; 


    // ORAM for graph
    struct OramConfig config(
        md.block_size, // block size
        md.bucket_size, // bucket size (Z)
        ntotal// num blocks
    );

    Bucket::setMaxSize(md.bucket_size);
    SBucket::setCipherSize(compute_ctx_len(md.bucket_size * (1 + md.block_size)*sizeof(int)));

    cout << "block_size: " << md.block_size << endl;
    cout << "bucket_size: " << md.bucket_size << endl;
    cout << "sbucket_size: " << SBucket::getCipherSize() << endl;

    RandForOramInterface* random = new RandomForOram();

    // build oram in plaintext
    size_t per_block_size = (1 + config.block_size);
    size_t per_bucket_size = config.bucket_size*per_block_size;
    size_t total_size = config.num_buckets * per_bucket_size;

    int* bkts = new int[total_size];
    // Initialize each block id to be -1
    for(size_t bucket_id = 0; bucket_id < config.num_buckets; bucket_id ++){
        for(size_t block_id = 0; block_id < config.bucket_size; block_id++){
            size_t offset = bucket_id*per_bucket_size + block_id*per_block_size;
            bkts[offset] = -1;
        }
    }

    cout << "Initializing..." << endl;

    int keyword_size = doc_scores.size();
    doc_to_position_map.resize(keyword_size);

    int cnt = 0;
    for(int kid = 0; kid < keyword_size; kid++){
        int doc_size = doc_scores[kid].size();
        for(int poffset = 0; poffset < doc_size; poffset++){
            int pid = doc_ids[kid][poffset];
            // Block b(rss->block_size);
            // b.index = to_position_map[kid][poffset];
            // b.data[0] = kid;
            // b.data[1] = pid;
            // ((float*)(b.data.data()))[2] = doc_scores[kid][poffset];
            doc_to_position_map[kid].push_back(cnt);

            bool written = false;
            while(!written){
                int leaf = random->getRandomLeafWithBound(config.num_leaves);
                vector<Block> blocks;
                for (int l = config.num_levels - 1; l >= 0; l--) {
                    size_t offset = (1<<l) - 1 + (leaf >> (config.num_levels - l - 1));
                    int* bkt = bkts + offset * per_bucket_size;
                    for(int block_id = 0; block_id < config.bucket_size; block_id++){
                        int* block = bkt + block_id * per_block_size;
                        if(block[0] == -1){
                            block[0] = cnt;
                            block[1] = kid;
                            block[2] = pid;
                            ((float*)block)[3] = doc_scores[kid][poffset];
                            // cout << "Assigning "<< cnt << " kid: " << kid << " pid: " << pid << " score: " << doc_scores[kid][poffset] << " to leaf: " << leaf << " bucket:" << block_id << endl;
                            written = true;
                            break;
                        }
                    }
                    if(written){
                        position_map.push_back(leaf);
                        break;
                    }
                }
            }
            cnt++;

            if(cnt % 1000000 == 0){
                cout << "Loading " << cnt << endl;
            }
        }
    }

    std::vector<SBucket> buckets(config.num_buckets);
    size_t per_block_offset = per_block_size*sizeof(int);
    size_t per_bucket_offset = per_bucket_size*sizeof(int);

    // encrypt
	for(int i = 0; i < config.num_buckets; i++){
		unsigned char* payload = (unsigned char*)bkts + i*per_bucket_offset;
		// Encrypt and write
		int ctx_len = encrypt_wrapper(payload, per_bucket_offset, buckets[i].data);
        // cout << "ctx_len: " << ctx_len << endl;
        // cout << "sb cipher: " << SBucket::getCipherSize() << endl;
		assert(ctx_len == SBucket::getCipherSize());

		// this->buckets[i]->from_ptr(payload);
	}

    // save buckets
    FILE* f = fopen(md.buckets_path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "could not open %s\n", md.buckets_path.c_str());
        perror("");
        abort();
    }
	// Write capacity
	fwrite(&config.num_buckets, 1, sizeof(int), f);
	// Write SBucket size
	int sb_size = SBucket::getCipherSize();
	fwrite(&sb_size, 1, sizeof(int), f);

    bool integrity = false;
	fwrite(&integrity, 1, sizeof(bool), f);


    for (size_t i = 0; i < config.num_buckets; i++){
        fwrite(buckets[i].data, 1, SBucket::getCipherSize(), f);
		if(integrity){
			fwrite(buckets[i].hash, 32, sizeof(uint8_t), f);
		}
    }
    fclose(f);

    // save position map
    save_position_map(md.pos_map_path.c_str(), position_map);

    return 0;
}