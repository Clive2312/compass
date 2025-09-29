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
# include "OramReadPathEviction.h"
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


int party = 0;
int port = 8000;
string address = "127.0.0.1";
int truncate_size = 10000;
string dataset = "";
string f_latency = "";
string f_comm = "";
int n = 1;

int main(int argc, char **argv) {
    ArgMapping amap;
    amap.arg("r", party, "Role of party: Server = 1; Client = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("d", dataset, "Dataset: [trip, msmarco]");
    amap.arg("n", n, "# queries");
    amap.arg("ip", address, "IP Address of server");
    amap.arg("trunc", truncate_size, "truncation of docs");
    amap.arg("f_latency", f_latency, "Save latency");
    amap.arg("f_comm", f_comm, "Save communication");
    amap.parse(argc, argv);

    cout << ">>> Setting up..." << endl;
    cout << "-> Role: " << party << endl;
    cout << "-> Address: " << address << endl;
    cout << "-> Port: " << port << endl;
    cout << "<<<" << endl << endl;

    bool is_server = party == 1;
    NetIO* io = new NetIO(is_server ? nullptr : address.c_str(), port);
    NetIO* bf_io = new NetIO(is_server ? nullptr : address.c_str(), port+1);

    double t0 = elapsed();

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
    vector<vector<int>> doc_to_position_map;

    tprint("Start reading... \n", t0);

    read_score(md.tfidf_score_path.c_str(), doc_scores, doc_ids, ntotal, max_docs, md.keyword_len);

    cout << "ntotal: " << ntotal << endl; 

    // ORAM for graph
    struct OramConfig oconfig(
        md.block_size, // block size
        md.bucket_size, // bucket size (Z)
        ntotal// num blocks
    );

    Bucket::setMaxSize(md.bucket_size);
    SBucket::setCipherSize(compute_ctx_len(md.bucket_size * (1 + md.block_size)*sizeof(int)));

    cout << "block_size: " << md.block_size << endl;
    cout << "bucket_size: " << md.bucket_size << endl;
    cout << "sbucket_size: " << SBucket::getCipherSize() << endl;

    bool in_memory = true;
    bool integrity = false;
    RemotePath* remote_storage = new RemotePath(io, oconfig, is_server, in_memory, integrity);

    RandForOramInterface* random = new RandomForOram();

    long comm;
    long round;

    if(is_server){
        // Config remote storage server
        tprint("Starting remote server... \n", t0);

        tprint("Starting loading... \n", t0);

        remote_storage->load_server_state(md.buckets_path.c_str());

        tprint("Done loading... \n", t0);
        bool ready;
        bf_io->send_data(&ready, sizeof(bool));

        tprint("Run server...\n", t0);

        comm = io->counter;
        round = io->num_rounds;
        
		remote_storage->run_server();

        if(f_comm != ""){
            long final_comm = io->counter - comm;
            io->send_data(&final_comm, sizeof(long));
        }
    } else{
        // Client
        tprint("Initializing ORAM... \n", t0);

        // rss->setCapacity(oconfig.num_buckets, false);

        OramInterface* oram = new OramReadPathEviction(
            remote_storage, 
            random, 
            oconfig.block_size,
            oconfig.bucket_size,
            oconfig.num_blocks
        );

        // loading queries
        int* xq;
        size_t nq;
        size_t dq;
        
        tprint("Loading queries... \n", t0);
        xq = ivecs_read(md.tfidf_query_path.c_str(), &dq, &nq);
        cout << "Loaded " << nq << "queries padded with " << dq << " words" << endl; 

        ((OramReadPathEviction*)oram)->loadPositionMapFromFile(md.pos_map_path.c_str());
        bool ready;
        bf_io->recv_data(&ready, sizeof(bool));

        // Create doc to position mapping
        int cnt = 0;
        doc_to_position_map.resize(doc_ids.size());
        for(int kid = 0; kid < doc_to_position_map.size(); kid++){
            int doc_size = doc_scores[kid].size();
            for(int poffset = 0; poffset < doc_size; poffset++){
                doc_to_position_map[kid].push_back(cnt);
                cnt++;
            }
        }

        vector<vector<int>> results;

        comm = io->counter;
        round = io->num_rounds;

        vector<float> latency;

        // Search
        {
            cout << "Searching " << endl;
            for(int i = 0; i < n; i++){
                double t1 = elapsed();
                // The ith query
                int* q = xq + i * dq;

                // cout << "Creating request: "  << i << endl;
                set<int> indices_set;
                vector<int> keywords;
                for(int j = 0; j < dq; j++){
                    // cout << "Creating request 1" << endl;
                    int kid = q[j];
                    if(kid > -1){
                        // cout << "Creating request 1.5" << endl;
                        int cnt = 0;
                        for(auto index : doc_to_position_map[kid]){
                            if(cnt >= truncate_size){
                                break;
                            }
                            // cout << "Creating request 2" << endl;
                            indices_set.insert(index);
                            // cout << "Getting "<< index << " kid: " << kid << endl;
                            // cout << "Creating request 3" << endl;
                            keywords.push_back(kid);
                            cnt ++;
                        }
                    }
                }

                // cout << "Sending bacth oram request " << endl;
                vector<int> indices_list(indices_set.begin(), indices_set.end());
                int useful_length = indices_list.size();
                int request_length = truncate_size * dq;
                // cout << "max_docs " << max_docs << endl;
                // cout << "dq " << dq << endl;
                // cout << "request_length "  << request_length << endl;
                // Pad to the same query
                for(int i = indices_list.size(); i < request_length; i++){
                    indices_list.push_back(0);
                }

                assert(request_length == indices_list.size());
                
                // indices.resize(request_length);
                vector<OramInterface::Operation> ops(request_length, OramInterface::Operation::READ);

                // query the oram
                vector<int*> accessed = ((OramReadPathEviction*)oram)->batch_multi_access_swap_ro(
                    ops,
                    indices_list,
                    vector<int*>(request_length, NULL),
                    indices_list.size()
                );

                if(useful_length > 0) {
                    map<int, vector<pair<int, float>>> key_doc_score;
                    for(int j = 0; j < useful_length; j++){
                        int* block = accessed[j];
                        int kid = block[0];
                        int pid = block[1];
                        float score = ((float*)block)[2];
                        key_doc_score[kid].push_back(make_pair(pid, score));
                    }

                    map<int, float> local_doc_scores;
                    for(int j = 0; j < dq; j++){
                        int kid = q[j];
                        if(kid > -1){
                        for(auto& pair: key_doc_score[kid]){
                                int pid = pair.first;
                                float score = pair.second;
                                auto it = local_doc_scores.find(pid);
                                if(it == local_doc_scores.end()){
                                    local_doc_scores[pid] = score;
                                } else{
                                    local_doc_scores[pid] = local_doc_scores[pid] + score;
                                }
                        }
                        }
                    }

                    vector<pair<int, float>> paired;
                    for(auto& pair : local_doc_scores){
                        paired.push_back({pair.first, pair.second});
                    }

                    std::sort(paired.begin(), paired.end(), 
                        [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                        return a.second > b.second;
                    });

                    vector<int> res;
                    for(int i = 0; i < k ; i++){
                        res.push_back(paired[i].first);
                    }
                    results.push_back(res);

                    // cout << "---> Result ";
                    // for(int i = 0; i < 3; i++){
                    //     cout << paired[i].first << " ";
                    // }
                    // cout << endl;

                    // cout << "---> Score ";
                    // for(int i = 0; i < 3; i++){
                    //     cout << paired[i].second << " ";
                    // }
                    // cout << endl;
                } else{

                    // cout << "---> Result ";
                    // for(int i = 0; i < 3; i++){
                    //     cout << 0 << " ";
                    // }
                    // cout << endl;

                    // cout << "---> Score ";
                    // for(int i = 0; i < 3; i++){
                    //     cout << 0 << " ";
                    // }
                    // cout << endl;
                    results.push_back(vector<int>(k, 0));
                }

                ((OramReadPathEviction*)oram)->evict_and_write_back();
                latency.push_back(elapsed() - t1);
                // printf("[%.3f s] Search time: %.3f s\n", elapsed() - t0, elapsed() - t1);
                cout << "-> " << i << endl;
            }

        }        

        if(f_latency != ""){
            fvecs_write(
                f_latency.c_str(),
                latency.data(),
                latency.size(),
                1
            );
        }

        remote_storage->close_server();

        if(f_comm != ""){
            long final_comm = io->counter - comm;
            long final_rnd = io->num_rounds - round;
            long server_comm;
            io->recv_data(&server_comm, sizeof(long));

            FILE* file = fopen(f_comm.c_str(), "w");
            if (file != nullptr) {
                fprintf(file, "%ld %ld\n", final_comm+server_comm, final_rnd);
                fclose(file);
            } else {
                perror("Error opening file");
            }
        }

    }

    std::cout << "Communication cost: " << io->counter - comm << std::endl;
    std::cout << "Round: " << io->num_rounds - round << std::endl;

    return 0;
}