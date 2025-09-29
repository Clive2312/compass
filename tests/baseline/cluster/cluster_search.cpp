#include "cluster_search.h"
#include "net_io_channel.h"
#include "ArgMapping.h"
#include "cluster_config_parser.h"

#include <chrono>
#include <omp.h>
#include <map>
#include <sys/stat.h>
#include <cfloat>
#include <faiss/index_io.h>
#include <faiss/IndexFlat.h>

using namespace std;
using namespace seal;

int party = 0;
int port = 8000;
string address = "127.0.0.1";
string dataset = "";
string f_latency = "";
string f_comm = "";
int scale = 1000;

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
    float* x = new float[n * (size_t)d];
    for (size_t i = 0; i < n; i++) {
        fread(&d, 1, sizeof(int), f);
        fread(x + i * (size_t)d, d, sizeof(float), f);
    }
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

// Function to load assignment mapping from file
map<int, vector<int>> load_assignment(const string& assignment_path) {
    map<int, vector<int>> assignment;

    FILE* f = fopen(assignment_path.c_str(), "rb");
    if (!f) {
        cerr << "Could not open assignment file: " << assignment_path << endl;
        return assignment;
    }

    int num_clusters;
    fread(&num_clusters, sizeof(int), 1, f);

    for (int i = 0; i < num_clusters; i++) {
        int cluster_id;
        int cluster_size;

        fread(&cluster_id, sizeof(int), 1, f);
        fread(&cluster_size, sizeof(int), 1, f);

        vector<int> nodes(cluster_size);
        fread(nodes.data(), sizeof(int), cluster_size, f);

        assignment[cluster_id] = nodes;
    }

    fclose(f);
    return assignment;
}

uint64_t fixed_point_encode(float a, uint64_t ptx_mod, uint64_t scale){
    double scaled = (double)scale * (double)a;
    int64_t q = (int64_t) llround(scaled);
    int64_t r = q % (int64_t)ptx_mod;
    if (r < 0) r += (int64_t)ptx_mod;
    return r;
}

float fixed_point_decode(uint64_t encoded, uint64_t ptx_mod, uint64_t scale) {
    // Center residue
    int64_t half = ptx_mod >> 1;
    int64_t centered = (encoded <= (uint64_t)half)
                       ? (int64_t)encoded
                       : (int64_t)encoded - (int64_t)ptx_mod;

    // Divide by scale^2 (since we multiplied two encoded values)
    double val = (double)centered / ((double)scale * (double)scale);
    return static_cast<float>(val);
}

void send_ciphertext(NetIO *io, Ciphertext &ct) {
  stringstream os;
  uint64_t ct_size;
  ct.save(os);
  ct_size = os.tellp();
  string ct_ser = os.str();
  io->send_data(&ct_size, sizeof(uint64_t));
  io->send_data(ct_ser.c_str(), ct_ser.size());
}

void recv_ciphertext(NetIO *io, Ciphertext &ct, SEALContext& context) {
  stringstream is;
  uint64_t ct_size;
  io->recv_data(&ct_size, sizeof(uint64_t));
  char *c_enc_result = new char[ct_size];
  io->recv_data(c_enc_result, ct_size);
  is.write(c_enc_result, ct_size);
  ct.unsafe_load(context, is);
  delete[] c_enc_result;
}

void send_encrypted_vector(NetIO *io, vector<Ciphertext> &ct_vec) {
    assert(ct_vec.size() > 0);
    stringstream os;
    uint64_t ct_size;
    for (size_t ct = 0; ct < ct_vec.size(); ct++) {
        ct_vec[ct].save(os, compr_mode_type::none);
        if (!ct)
        ct_size = os.tellp();
    }
    string ct_ser = os.str();
    io->send_data(&ct_size, sizeof(uint64_t));
    io->send_data(ct_ser.c_str(), ct_ser.size());
    // for(auto& ctx : ct_vec){
    //     send_ciphertext(io, ctx);
    // }
}

void recv_encrypted_vector(NetIO *io, vector<Ciphertext> &ct_vec, SEALContext& context) {
    assert(ct_vec.size() > 0);
    stringstream is;
    uint64_t ct_size;
    io->recv_data(&ct_size, sizeof(uint64_t));
    char *c_enc_result = new char[ct_size * ct_vec.size()];
    io->recv_data(c_enc_result, ct_size * ct_vec.size());
    for (size_t ct = 0; ct < ct_vec.size(); ct++) {
        is.write(c_enc_result + ct_size * ct, ct_size);
        ct_vec[ct].unsafe_load(context, is);
    }
    delete[] c_enc_result;
//     for(auto& ctx : ct_vec){
//         recv_ciphertext(io, ctx, context);
//     }
}

inline double interval(chrono::_V2::system_clock::time_point start){
    auto end = std::chrono::high_resolution_clock::now();
    auto interval = (end - start)/1e+9;
    return interval.count();
}



int main(int argc, char **argv){

    ArgMapping amap;
    amap.arg("r", party, "Role of party: Server = 1; Client = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("ip", address, "IP Address of server");
    amap.arg("d", dataset, "Dataset: [sift, trip, laion]");
    amap.arg("f_latency", f_latency, "Save latency");
    amap.arg("f_comm", f_comm, "Save communication");

    amap.parse(argc, argv);

    if (dataset.empty()) {
        cerr << "Error: Dataset must be specified with -d option" << endl;
        cerr << "Available datasets: sift, trip, laion" << endl;
        return 1;
    }

    // Parse configuration from JSON
    ClusterMetadata md;
    cout << "-> Parsing json config..." << endl;
    if (parseClusterJson("../tests/baseline/cluster/clustering/config.json", md, dataset) != 0) {
        cerr << "Failed to parse configuration. Exiting..." << endl;
        return 1;
    }

    // Extract configuration values
    int num_queries = md.num_queries;
    size_t slot_count = md.slot_count;
    int k = md.k;
    int nc = md.nc;
    int dim = md.dim;
    int node_per_cluster = md.node_per_cluster;

    cout << ">>> Setting up..." << endl;
    cout << "-> Role: " << party << endl;
    cout << "-> Address: " << address << endl;
    cout << "-> Port: " << port << endl;
    cout << "-> Dataset: " << dataset << endl;
    cout << "-> Num clusters: " << nc << endl;
    cout << "-> Dimension: " << dim << endl;
    cout << "-> Nodes per cluster: " << node_per_cluster << endl;
    cout << "<<<" << endl << endl;

    bool isServer = party == 1;
    NetIO* io = new NetIO(isServer ? nullptr : address.c_str(), port);

    // Initialize HE Params

    EncryptionParameters parms(scheme_type::bfv);
    size_t poly_modulus_degree = slot_count;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
    parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 37));

    SEALContext context(parms);
    print_parameters(context);
    cout << endl;

    Evaluator evaluator(context);

    // Encoder
    BatchEncoder batch_encoder(context);

    long comm;
    long round;

    // // encoding test
    // uint64_t ptx_mod = parms.plain_modulus().value();
    // float a = 0.3377f;
    // float b = 0.4455f;

    // // encode
    // uint64_t enc_a = fixed_point_encode(a, ptx_mod, scale);
    // uint64_t enc_b = fixed_point_encode(b, ptx_mod, scale);

    // // simulate homomorphic multiplication: multiply encoded ints modulo ptx_mod
    // uint64_t enc_prod = (enc_a * enc_b) % (uint64_t)ptx_mod;

    // // decode
    // float approx = fixed_point_decode(enc_prod, ptx_mod, scale);

    // // reference
    // float ref = a * b;

    // std::cout << "a=" << a << " b=" << b << "\n";
    // std::cout << "encoded a=" << enc_a << " encoded b=" << enc_b << "\n";
    // std::cout << "encoded product=" << enc_prod << "\n";
    // std::cout << "decoded product=" << approx << "\n";
    // std::cout << "reference product=" << ref << "\n";

    // return 0;
              

    if(isServer){

        // Recieve keys 
        cout << "Waiting for keys from client ..." << endl;
        PublicKey pub_key;
        SecretKey sec_key;
        RelinKeys relin_keys;

        // Public key
        {
            cout << " - Pub Key" << endl;
            uint64_t pk_size;
            io->recv_data(&pk_size, sizeof(uint64_t));
            char *key_share = new char[pk_size];
            io->recv_data(key_share, pk_size);
            stringstream is;
            is.write(key_share, pk_size);
            pub_key.load(context, is);
            delete[] key_share;
        }
        

        // Secret key
        // {
        //     cout << " - Sec Key" << endl;
        //     uint64_t sk_size;
        //     io->recv_data(&sk_size, sizeof(uint64_t));
        //     char *key_share_sk = new char[sk_size];
        //     io->recv_data(key_share_sk, sk_size);
        //     stringstream is_sk;
        //     SecretKey sec_key;
        //     is_sk.write(key_share_sk, sk_size);
        //     sec_key.load(context, is_sk);
        //     delete[] key_share_sk;
        // }
		

        // Relin key
        {
            cout << " - Relin Key" << endl;
            uint64_t relin_size;
            io->recv_data(&relin_size, sizeof(uint64_t));
            char *key_share_relin = new char[relin_size];
            io->recv_data(key_share_relin, relin_size);
            stringstream is_relin;
            is_relin.write(key_share_relin, relin_size);
            relin_keys.load(context, is_relin);
            delete[] key_share_relin;
        }

        Encryptor encryptor(context, pub_key);
        // encryptor.set_secret_key(sec_key);
		// Decryptor decryptor(context, sec_key);
		

        // Prepare database
        cout << "Preparing database..." << endl;

        // Load the base vectors
        size_t d_base, n_base;
        cout << "Loading base vectors from: " << md.base_path << endl;
        float* base_vectors = fvecs_read(md.base_path.c_str(), &d_base, &n_base);
        assert(d_base == dim);

        // Print first 2 embeddings for verification
        // // Find min and max values in the entire base_vectors
        // float min_val = FLT_MAX;
        // float max_val = -FLT_MAX;
        // for(size_t i = 0; i < n_base * d_base; i++){
        //     if(base_vectors[i] < min_val) min_val = base_vectors[i];
        //     if(base_vectors[i] > max_val) max_val = base_vectors[i];
        // }
        // cout << "Base vectors statistics:" << endl;
        // cout << "  Min value: " << min_val << endl;
        // cout << "  Max value: " << max_val << endl;
        // cout << "  Range: " << (max_val - min_val) << endl;

        // return 0;
        // assert(0);

        // Load the assignment mapping
        cout << "Loading assignment from: " << md.assignment_path << endl;
        map<int, vector<int>> assignment = load_assignment(md.assignment_path);
        cout << "Loaded assignment with " << assignment.size() << " clusters" << endl;

        vector<vector<Ciphertext>> db;
        int num_embed_per_ctx = slot_count / dim;
        int num_ctx_per_cluster = (node_per_cluster / num_embed_per_ctx) + 1;

        cout << "- num_embed_per_ctx: " << num_embed_per_ctx << endl;
        cout << "- num_ctx_per_cluster:" << num_ctx_per_cluster << endl;
        db.resize(nc);

        #pragma omp parallel for
        for(int cid = 0; cid < nc; cid++){
            // Based on the actual assignment, load the data into pod accordingly
            // cout << "encrypting cluster: " << cid << endl;
            vector<int> cluster_nodes;
            if (assignment.find(cid) != assignment.end()) {
                cluster_nodes = assignment[cid];
            } else{
                assert(0);
            }

            int node_idx = 0;
            for(int i = 0; i < num_ctx_per_cluster; i++){
                vector<uint64_t> pod(slot_count, 0ULL);

                // Pack embeddings into this ciphertext
                for(int j = 0; j < num_embed_per_ctx && node_idx < cluster_nodes.size(); j++){
                    int node_id = cluster_nodes[node_idx];

                    // Copy the embedding data for this node
                    for(int k = 0; k < dim; k++){
                        // Convert float to uint64_t (you may need to adjust this conversion)
                        float val = base_vectors[node_id * dim + k];
                        // Simple conversion - you might need a better encoding scheme
                        pod[j * dim + k] = fixed_point_encode(val, parms.plain_modulus().value(), scale);
                    }
                    node_idx++;
                }

                Plaintext ptx;
                Ciphertext ctx;
                batch_encoder.encode(pod, ptx);
                encryptor.encrypt(ptx, ctx);
                db[cid].push_back(ctx);
            }
        }

        delete[] base_vectors;

        // Server is ready
        cout << "Server is ready..." << endl;
        bool signal = true;
        io->send_data(&signal, sizeof(bool));

        comm = io->counter;
        round = io->num_rounds;

        for(int iq = 0; iq < num_queries; iq++){
            vector<Ciphertext> query_ctxs(nc);
            vector<Ciphertext> results_ctxs(num_ctx_per_cluster);

            // recv ctx from server
            cout << "Waiting for query..." << endl;
            recv_encrypted_vector(io, query_ctxs, context);
            cout << "Received query from client..." << endl;

            // auto t_start = std::chrono::high_resolution_clock::now();

            // compute
            #pragma omp parallel for
            for(int i = 0; i < num_ctx_per_cluster; i++){
                Ciphertext sum;
                vector<Ciphertext> tmp_mult;
                for(int cid = 0; cid < nc; cid++){
                    Ciphertext mult;
                    evaluator.multiply(db[cid][i], query_ctxs[cid], mult);
                    evaluator.relinearize_inplace(mult, relin_keys);
                    tmp_mult.push_back(mult);
                    if(cid == 0){
                        sum = mult;
                    } else{
                        evaluator.add_inplace(sum, mult);
                    }
                }

                results_ctxs[i] = sum;
            }

            // cout << "> [TIMING]: server computation: " << interval(t_start) << "sec" << endl;

            cout << "Sending query result back to client..." << endl;
            send_encrypted_vector(io, results_ctxs);
        }

        if(f_comm != ""){
            long final_comm = io->counter - comm;
            io->send_data(&final_comm, sizeof(long));
        }

    } else{


        // Generate Keys
        cout << "Generating keys..." << endl;
        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        PublicKey public_key;
        keygen.create_public_key(public_key);
        RelinKeys relin_keys;
        keygen.create_relin_keys(relin_keys);
        Encryptor encryptor(context, public_key);
        Decryptor decryptor(context, secret_key);

        // Share the keys 
        cout << "Sharing keys with server ..." << endl;
        {
            stringstream os;
            public_key.save(os);
            uint64_t pk_size = os.tellp();
            string keys_ser = os.str();
            io->send_data(&pk_size, sizeof(uint64_t));
            io->send_data(keys_ser.c_str(), pk_size);
        }

        // {
        //     stringstream os_sk;
        //     secret_key.save(os_sk);
        //     uint64_t sk_size = os_sk.tellp();
        //     string keys_ser_sk = os_sk.str();
        //     io->send_data(&sk_size, sizeof(uint64_t));
        //     io->send_data(keys_ser_sk.c_str(), sk_size);
        // }

        {
            stringstream os_relin;
            relin_keys.save(os_relin);
            uint64_t relin_size = os_relin.tellp();
            string keys_ser_relin = os_relin.str();
            io->send_data(&relin_size, sizeof(uint64_t));
            io->send_data(keys_ser_relin.c_str(), relin_size);

        }

        // Read Queries
        size_t nq;
        float* xq;
        size_t d2;

        cout << "Loading queries from: " << md.query_path << endl;
        xq = fvecs_read(md.query_path.c_str(), &d2, &nq);
        assert(dim == d2);
        cout << "Loaded " << nq << " queries of dimension " << d2 << endl;

        // Load centroid index for determining cluster
        cout << "Loading centroid index from: " << md.centroid_path << endl;
        faiss::Index* centroid_index = faiss::read_index(md.centroid_path.c_str());
        cout << "Loaded centroid index with " << centroid_index->ntotal << " centroids" << endl;

        int num_embed_per_ctx = slot_count / dim;
        int num_ctx_per_cluster = (node_per_cluster / num_embed_per_ctx) + 1;

        // return 0;

        // Waiting for server to be ready
        cout << "Waiting for server to be ready..." << endl;
        bool signal;
        io->recv_data(&signal, sizeof(bool));

        comm = io->counter;
        round = io->num_rounds;

        vector<vector<int>> search_results;

        auto t_start = std::chrono::high_resolution_clock::now();

        // Perform Search
        for(int i = 0; i < num_queries; i++){
            cout << "Performing search: " << i << endl;

            // Determine cluster by finding nearest centroid
            float* query = xq + i * dim;
            faiss::idx_t* cluster_id = new faiss::idx_t[1];
            float* cluster_dist = new float[1];
            centroid_index->search(1, query, 1, cluster_dist, cluster_id);
            int cluster = cluster_id[0];
            cout << "Query " << i << " assigned to cluster " << cluster << endl;
            delete[] cluster_id;
            delete[] cluster_dist;

            // Generate query ctx
            cout << "Generating query ctx... " << endl;
            vector<Ciphertext> query_ctxs;
            for(int cid = 0; cid < nc; cid++){
                vector<uint64_t> pod(slot_count, 0ULL);

                if(cid == cluster){
                    // Encrypt the actual query vector
                    // Replicate query across all embedding slots for parallel computation
                    for(int j = 0; j < num_embed_per_ctx; j++){
                        for(int k = 0; k < dim; k++){
                            // Convert float to uint64_t (matching server side encoding)
                            float val = query[k];
                            pod[j * dim + k] = fixed_point_encode(val, parms.plain_modulus().value(), scale);
                        }
                    }
                }
                // For non-matching clusters, pod remains all zeros

                Plaintext ptx;
                Ciphertext ctx;
                batch_encoder.encode(pod, ptx);
                encryptor.encrypt(ptx, ctx);
                query_ctxs.push_back(ctx);
            }


            // Send ctx to server
            assert(query_ctxs.size() == nc);
            cout << "Sending query ctx... " << endl;
            send_encrypted_vector(io, query_ctxs);

           
            // Recv ctx from server
            cout << "Waiting for query result... " << endl;
            vector<Ciphertext> result_ctxs(num_ctx_per_cluster);
            recv_encrypted_vector(io, result_ctxs, context);
            cout << " - Recieved!" << endl;

            vector<float> results;
            int cnt = 0;

            // Decrypt and compute sum
            cout << "Computing sum... " << endl;
            for(auto ctx : result_ctxs){
                Plaintext ptx;
                vector<uint64_t> pod(slot_count, 0ULL);
                decryptor.decrypt(ctx, ptx);
                batch_encoder.decode(ptx, pod);

                for(int j = 0; j < num_embed_per_ctx; j++){
                    float result = 0.0;
                    for(int k = 0; k < d2; k++){
                        // Convert back from uint64_t to float (reverse of encoding)
                        float val = fixed_point_decode(pod[j * d2 + k], parms.plain_modulus().value(), scale);
                        
                        result += val;
                    }
                    // Store as scaled integer for sorting (or you can keep as float)
                    results.push_back(result);
                    cnt ++;
                    if(cnt > node_per_cluster){
                        break;
                    }
                }

                if(cnt > node_per_cluster){
                    break;
                }
            }

            assert(results.size() == node_per_cluster);

            bool verification = false;
            if(verification){
                // Load base vectors for verification
                size_t d_base_verify, n_base_verify;
                cout << "Loading base vectors for verification from: " << md.base_path << endl;
                float* base_vectors = fvecs_read(md.base_path.c_str(), &d_base_verify, &n_base_verify);
                assert(d_base_verify == dim);

                // Load assignment for verification
                cout << "Loading assignment for verification from: " << md.assignment_path << endl;
                map<int, vector<int>> assignment = load_assignment(md.assignment_path);
                cout << "Loaded assignment with " << assignment.size() << " clusters for verification" << endl;

                cout << "Computing plaintext verification..." << endl;
                vector<float> plaintext_results;

                // Load assignment for this cluster to know which nodes to compute
                if (assignment.find(cluster) != assignment.end()) {
                    vector<int> cluster_nodes = assignment[cluster];

                    // Compute dot products for each node in the cluster
                    for (int node_id : cluster_nodes) {
                        float dot_product = 0.0;
                        for (int k = 0; k < dim; k++) {
                            float base_val = base_vectors[node_id * dim + k];
                            float query_val = query[k];
                            dot_product += base_val * query_val;
                        }
                        plaintext_results.push_back(dot_product);
                    }

                } else {
                    cout << "  WARNING: Cluster " << cluster << " not found in assignment!" << endl;
                    assert(0);
                    // Fill with zeros if cluster not found
                    for(int j = 0; j < node_per_cluster; j++) {
                        plaintext_results.push_back(0.0);
                    }
                }

                cout << "Verifying correctness..." << endl;
                bool verification_passed = true;
                float max_error = 0.0;
                int num_to_verify = min((int)plaintext_results.size(), (int)results.size());

                for(int j = 0; j < num_to_verify; j++){
                    float encrypted_result = results[j];
                    float plaintext_result = plaintext_results[j];
                    float error = abs(encrypted_result - plaintext_result);
                    max_error = max(max_error, error);

                    // Allow some tolerance for floating point errors and encoding/decoding
                    if(error > 0.1) {
                        cout << "  WARNING: Mismatch at index " << j << endl;
                        cout << "    Encrypted: " << encrypted_result << endl;
                        cout << "    Plaintext: " << plaintext_result << endl;
                        cout << "    Error: " << error << endl;
                        verification_passed = false;
                    }
                }

                if(verification_passed) {
                    cout << "Verification PASSED (max error: " << max_error << ")" << endl;
                } else {
                    cout << "Verification FAILED" << endl;
                }

                delete[] base_vectors;
            }

            // Sort
            cout << "Ranking... " << endl;
            vector<int> arg;

            for(int i = 0; i < node_per_cluster; i++){
                arg.push_back(i);
            }

            std::sort(arg.begin(), arg.end(), [&results](int a, int b) {
                return results[a] < results[b];
            });

            search_results.push_back(vector<int>(arg.begin(), arg.begin() + k));
        }

        if(f_latency != ""){
            // string perceived_latency_path = "perceived_latency_" + dataset + ".bin";
            // string full_latency_path = "full_latency_" + dataset + ".bin";

            float lat = interval(t_start);

            vector<float> latency = {lat};

            fvecs_write(
                f_latency.c_str(),
                latency.data(),
                latency.size(),
                1
            );

        }

        cout << "> [TIMING]: server computation: " << interval(t_start) << "sec" << endl;

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

        // Clean up
        delete[] xq;
        delete centroid_index;
    }

   

    std::cout << "Communication cost: " << io->counter - comm << std::endl;
    std::cout << "Round: " << io->num_rounds - round << std::endl;

    return 0;
}