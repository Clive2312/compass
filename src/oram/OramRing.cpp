//
//
//

#include "OramRing.h"
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <cassert>
#include <chrono>
#include <algorithm>
#include <random>

inline double interval(std::chrono::_V2::system_clock::time_point start){
    auto end = std::chrono::high_resolution_clock::now();
    auto interval = (end - start)/1e+9;
    return interval.count();
}

int OramMetaData::GetBlockOffset(int block_id, bool& real){
    int ret = -1;

    bool dummy_found = false;
    int dummy_ret = -1;

    // This implementation is insecure
    // Should randomly sample from a dummy instead of linear traversal

    for(int i = 0; i < block_ids.size(); i++){
        if(block_id == block_ids[i] && valids[i] == 1){
            ret = i;
            real = true;
            valids[i] = 0;
            return ret;
        }
        if(!dummy_found){
            if(block_ids[i] == -1 && valids[i] == 1){
                dummy_ret = i;
                // TODO: insecure workaround, always use first dummy
                dummy_found = true;
            }
        }
    }

    // cout << "GetBlockOffset for "<< block_id << " : ";
    // for(auto id : block_ids){
    //     cout << id << " ";
    // }
    // cout << endl;
    valids[dummy_ret] = 0;
    return dummy_ret;
}

int OramMetaData::GetBlockOffsetLeak(int block_id, bool& real){
    int ret = -1;

    bool dummy_found = false;
    int dummy_ret = -1;

    // This implementation is insecure
    // Should randomly sample from a dummy instead of linear traversal

    for(int i = 0; i < block_ids.size(); i++){
        if(block_id == block_ids[i] && valids[i] == 1){
            ret = i;
            real = true;
            // valids[i] = 0;
            return ret;
        }
        if(!dummy_found){
            if(block_ids[i] == -1 && valids[i] == 1){
                dummy_ret = i;
                // TODO: insecure workaround, always use first dummy
                dummy_found = true;
            }
        }
    }

    // cout << "GetBlockOffset for "<< block_id << " : ";
    // for(auto id : block_ids){
    //     cout << id << " ";
    // }
    // cout << endl;
    // valids[dummy_ret] = 0;
    return dummy_ret;
}

void OramMetaData::print_stat(){
    cout << "- count: " << (int) count << endl;
    cout << "- valids: " << valids << endl;
    cout << "- block_ids: ";
    for(auto bid : block_ids){
        cout << bid << " ";
    }
    cout << endl;
}

// int OramMetaData::ShuffleAndReset(){
//     // Generate shuffle mapping
//     vector<int> mapping(block_ids.size());
//     // Fill the mapping vector with sequential indices
//     for (int i = 0; i < block_ids.size(); ++i) {
//         mapping[i] = i;
//     }

//     // Shuffle the mapping vector
//     std::random_device rd;  // Seed
//     std::mt19937 gen(rd()); // Random number generator
//     std::shuffle(mapping.begin(), mapping.end(), gen);
// }


OramRing::OramRing(RemoteRing* storage, RandForOramInterface* rand_gen, RingOramConfig config, int num_levels, int cached_levels, bool batch, bool lazy) {
    this->storage = storage;
    this->rand_gen = rand_gen;
    this->block_size = config.block_size;
    this->real_bucket_size = config.real_bucket_size;
    this->dummy_size = config.dummy_size;
    this->bucket_size = config.real_bucket_size + config.dummy_size;
    this->evict_rate = config.evict_rate;
    this->num_blocks = config.num_blocks;

    this->batching = batch;
    this->lazy_eveiction = lazy;

    // this->num_levels = ceil(log10(2 * num_blocks / evict_rate) / log10(2)) - 1;
    this->num_levels = num_levels;
    this->num_buckets = pow(2, num_levels)-1;

    this->G = 1;
    
    if (this->num_buckets*this->bucket_size < this->num_blocks) //deal with precision loss
    {
        throw new runtime_error("Not enough space for the acutal number of blocks.");
    }
    this->num_leaves = pow(2, num_levels-1);
    this->rand_gen->setBound(num_leaves);
    // cout << "Set capacity" << endl; 
    // this->storage->setCapacity(num_buckets);
    // cout << "Set capacity done" << endl; 
    this->position_map = vector<int>(num_blocks);
    this->metadata = vector<OramMetaData>(num_buckets);

    // this->stash = vector<Block>();
    this->mstash = map<int, Block>();
    this->mmstash = map<std::pair<int, int>, Block*, PairComparator>();
    
    for (int i = 0; i < this->num_blocks; i++){
        position_map[i] = rand_gen->getRandomLeafWithBound(num_leaves);
    }

    cache_top = cached_levels > 0;
    num_cached_levels = cached_levels;

    if(cache_top){
        num_cached_bucket = pow(2, num_cached_levels)-1;
    }

    cnt_early_reshuffle = 0;
    cnt_delayed_leaves = 0;
}

void OramRing::init_cache_top(){

    // following code has an assumption that usually tree top is empty at the beginning
    vector<int> positions;
    vector<int> offsets;
    vector<bool> valids;
    vector<int> num_non_dummies;
    int total_non_dummies;
    for(int pos = 0; pos < num_cached_bucket; pos++){
        int non_dummies = 0;
        for(int i = 0; i < bucket_size; i++){
            if(metadata[pos].block_ids[i] != -1){
                // no need to check valids
                positions.push_back(pos);
                offsets.push_back(i);
                valids.push_back(true);
                non_dummies++;
            }
        }
        total_non_dummies += non_dummies;
        num_non_dummies.push_back(non_dummies);
        assert(non_dummies <= real_bucket_size);
    }

    if(total_non_dummies > 0) {
        assert(0);
    } else{
        return;
    }

    //retrieve
    std::vector<Block*> blocks;
    storage->ReadBlockBatchAsBlockRing(positions, offsets, blocks, valids, false);

    for(auto b : blocks){
        // cached_oram.insert({b->index, b});
        mmstash[make_pair(position_map[b->index], b->index)] = b;
    }

    cout << mmstash.size() << " blocks cached locally." << endl;

    // int cnt = 0;
    // for(int pos = 0; pos < num_cached_bucket; pos++){
    //     int non_dummies = num_non_dummies[pos];
    //     for(int i = 0; i < non_dummies; i++){
    //         cached_oram[pos*real_bucket_size + i] = blocks[cnt];
    //         cnt++;
    //     }
    //     for(int i = non_dummies; i < real_bucket_size; i++){
    //         cached_oram[pos*real_bucket_size + i] = new Block(block_size);
    //     }
    // }
}

int OramRing::ReverseBits(int g, int bits_length) {
    /*
    INPUT: Integers g and bits_length.
    OUTPUT: Integer reverse of length bits_length consisting of reversed bits of g.
    To be used to traverse the leaves and run eviction in reverse lexicographical order.
    */
    int g_mod = g%num_leaves;
    int reverse = 0;
    while(g_mod) {
        reverse <<= 1;
        reverse |= g_mod & 1;
        g_mod >>= 1;
        bits_length--;
    }
    
    reverse <<= bits_length;
    return reverse;
}

void OramRing::early_reshuffle(vector<int> buckets){
    
    vector<int> positions;
    vector<int> offsets;
    vector<bool> valids;

    for(int i = 0; i < buckets.size(); i++){
        // Fetch exactly Z blocks from each buckets
        int pos = buckets[i];
        int z = 0;
        for(int j = 0; j < bucket_size; j++){
            if(metadata[pos].block_ids[j] != -1 && metadata[pos].valids[j] == 1){
                // valid non-empty block
                positions.push_back(pos);
                offsets.push_back(j);
                valids.push_back(true);
                z++;
            }
        }

        // cout << "valid z: " << z << endl;

        for(int j = 0; j < bucket_size; j++){
            if(z >= real_bucket_size){
                break;
            }
            // cout << "j: " << j << endl;
            // cout << "metadata[pos].block_ids[j]: " << metadata[pos].block_ids[j] << endl;
            // cout << "metadata[pos].valids[j]: " << metadata[pos].valids[j] << endl;

            if(metadata[pos].block_ids[j] == -1 && metadata[pos].valids[j] == 1){
                // valid empty block
                positions.push_back(pos);
                offsets.push_back(j); 
                valids.push_back(true);
                z++;
            }
        }

        if(z < real_bucket_size){
            cout << "bucket size: " << bucket_size << endl;
            cout << "real_bucket_size: " << real_bucket_size << endl;
            cout << "z: " << z << endl;
            metadata[pos].print_stat();
            assert(0);
        }

        assert(z == real_bucket_size);

        // cout << "pos: " << pos << endl;
        // cout << "z: " << z << endl;

        assert(z == real_bucket_size);
    }

    std::vector<Block*> blocks;
    storage->ReadBlockBatchAsBlockRing(positions, offsets, blocks, valids, true);

    std::random_device rd;  
    std::mt19937 g(rd());
    std::map<int, vector<Block*>> reshuffled_buckets;
    for(int i = 0; i < buckets.size(); i++){
        int pos = buckets[i];
        std::vector<Block*> tmp_bkt(
            blocks.begin() + i * real_bucket_size, 
            blocks.begin() + (i + 1) * real_bucket_size
        ); 

        for(int i = 0; i < dummy_size; i++){
            Block* dummy = new Block(block_size);
            tmp_bkt.push_back(dummy); //dummy block
        }
        assert(tmp_bkt.size() == bucket_size);

        // shuffle
        std::shuffle(tmp_bkt.begin(), tmp_bkt.end(), g);

        // update metadata
        for(int j = 0; j < bucket_size; j++){
            metadata[pos].block_ids[j] = tmp_bkt[j]->index;
        }
        metadata[pos].count = 0;
        metadata[pos].valids.set();

        reshuffled_buckets[pos] = tmp_bkt;
    }

    // write back to server
    storage->WriteBucketBatchMapAsBlockRing(reshuffled_buckets, bucket_size, true);

}

std::vector<int*> OramRing::access(std::vector<Operation> ops, std::vector<int> blockIndices, std::vector<int*> newdata, int pad_l){
    if(batching){

        // a workaround for super large batch size
        int num_remote_roots = 1 << (num_cached_levels);
        int max_batch_size = num_remote_roots * dummy_size / 2;
        if( pad_l > max_batch_size){
            cout << "split large batch" << endl;
            vector<int*> res;
            int cnt_request = 0;
            int remain_requests = pad_l;
            while(remain_requests > 0){
                int tmp_batch_size = (remain_requests > max_batch_size) ? max_batch_size : remain_requests;
                std::vector<Operation> tmp_ops = {};
                std::vector<int> tmp_block_indices = {};
                std::vector<int*> tmp_new_data = {};
                if(ops.size() >= cnt_request + tmp_batch_size){
                    // all real request
                    tmp_ops = std::vector<Operation>(ops.begin() + cnt_request, ops.begin() + tmp_batch_size);
                    tmp_block_indices = std::vector<int>(blockIndices.begin() + cnt_request, blockIndices.begin() + tmp_batch_size);
                    tmp_new_data = std::vector<int*>(newdata.begin() + cnt_request, newdata.begin() + tmp_batch_size);
                } else if (ops.size() > cnt_request){
                    // partial real request
                    tmp_ops = std::vector<Operation>(ops.begin() + cnt_request, ops.end());
                    tmp_block_indices = std::vector<int>(blockIndices.begin() + cnt_request, blockIndices.end());
                    tmp_new_data = std::vector<int*>(newdata.begin() + cnt_request, newdata.end());
                }
                // other wise empty is fine

                vector<int*> tmp_result = batch_multi_access_swap_ro(
                    tmp_ops,
                    tmp_block_indices,
                    tmp_new_data,
                    tmp_batch_size
                );
                res.insert(res.end(), tmp_result.begin(), tmp_result.end());
                cnt_request += tmp_batch_size;
                remain_requests -= tmp_batch_size;
            }

            return res;
        } else{
            return batch_multi_access_swap_ro(ops, blockIndices, newdata, pad_l);
        }

        return batch_multi_access_swap_ro(ops, blockIndices, newdata, pad_l);
    } else{
        vector<int*> ret;
        for(int i = 0; i < ops.size(); i++){
            vector<int*> tmp = batch_multi_access_swap_ro(
                {ops[i]},
                {blockIndices[i]},
                {newdata[i]},
                1
            );
            ret.push_back(tmp[0]);
        }
        for(int i = ops.size(); i < pad_l; i++){
            batch_multi_access_swap_ro(
                {},
                {},
                {},
                1
            );
        }
        return ret;
    }
}


/*
    Workflow here:
    1. Find all the leaves
    2. Check the meta data to see if there's any bucket needs reshuffle
    3. Early reshuffle them
    4. Fetch the path
*/


std::vector<int*> OramRing::batch_multi_access_swap_ro(std::vector<Operation> ops, std::vector<int> blockIndices, std::vector<int*> newdata , int pad_l){
    std::vector<int*> ret;

    // check each bucket on the paths to see if early reshuffle is required
    {
        vector<int> leaves; // old leaves
        vector<int> positions; // bucket offsets
        vector<int> block_ids; // only for reshuflle

        bool reshuffle = false;
        set<int> reshuffle_buckets;

        // check for real access
        for(int block_id : blockIndices){
            int oldLeaf = position_map[block_id];
            bool found = false;

            // traverse the path
            for (int l = num_levels - 1; l >= num_cached_levels; l--) {
                int pos = P(oldLeaf, l);
                // check metadata and increase count
                this->metadata[pos].count++;

                if(this->metadata[pos].count >= dummy_size){
                    cnt_early_reshuffle += 1;
                    reshuffle = true;
                    reshuffle_buckets.insert(pos);
                }

                block_ids.push_back(block_id);
                positions.push_back(pos);
            }

            leaves.push_back(oldLeaf);
        }

        // check for dummy access
        for(int i = leaves.size(); i < pad_l; i++){
            int newLeaf = rand_gen->getRandomLeafWithBound(num_leaves);

            for (int l = num_levels - 1; l >= num_cached_levels; l--) {
                int pos = P(newLeaf, l);

                // check metadata and increase count
                this->metadata[pos].count++;

                if(this->metadata[pos].count >= dummy_size){
                    cnt_early_reshuffle += 1;
                    reshuffle = true;
                    reshuffle_buckets.insert(pos);
                }
                block_ids.push_back(-1);
                positions.push_back(pos);
            }
        }

        // early_reshuffle({65530, 5677});

        // reshuffle if necessary
        if(reshuffle){
            cout << "Early reshuffle " << reshuffle_buckets.size() << " buckets." << endl;
            early_reshuffle(vector<int>(reshuffle_buckets.begin(), reshuffle_buckets.end()));
        }

        // generate the block offset for each bucket
        vector<int> offsets; // interested block offset within each bucket
        vector<bool> valids; // whether the block is a useful block, each path should contain exactly 1 useful block (could be a dummy tho)

        for(int i = 0; i < positions.size(); i++){
            int pos = positions[i];
            bool real = false;
            int offset = metadata[pos].GetBlockOffset(block_ids[i], real);
            offsets.push_back(offset);
            if(block_ids[i] != -1){
                valids.push_back(real);
            } else{
                valids.push_back(false);
            }

            if(reshuffle && reshuffle_buckets.find(pos) != reshuffle_buckets.end()){
                // update the count if this bucket is reshuffled
                metadata[pos].count++;
                if(metadata[pos].count >= dummy_size){
                    // this should not happen statistically
                    assert(0);
                }
            }
        }

        // revisit valids incase of cache miss
        int interval  = num_levels - num_cached_levels;
        assert(valids.size() == interval * pad_l);
        for(int i = 0; i < pad_l; i++){
            bool found = false;
            for (int j = 0;j < interval; j++) {
                found = found || valids[i*interval + j];
            }
            // mark the last dummy as valid block
            if(!found){
                valids[i*interval] = true;
            }
        }

        // update the metadata
        // for(int i = 0; i < positions.size(); i++){
        //     int pos = positions[i];
        //     int offset = offsets[i];
        //     metadata[pos] 
        // }

        // cout << "positions and offsets generated" << endl;

        cnt_delayed_leaves += pad_l;

        // cout << "cnt_delayed_leaves: " << cnt_delayed_leaves << endl;

        // Batch read
        std::vector<Block*> blocks;
        // auto t_0 = std::chrono::high_resolution_clock::now();
        storage->ReadBlockBatchAsBlockRingXor(positions, offsets, blocks, pad_l, valids);

        // cout << "saving data" << endl;

        for(Block* b : blocks){
            int idx = b->index;
            // cout << "saving data: " << idx << endl;
            if (idx != -1) {
                // cout << "Got: " << b->index << endl;
                if(mmstash.find(make_pair(position_map[idx], idx)) != mmstash.end()){
                    cout << "XXXXXXXXXXXXXXXXXXXXXXXX Overwrite mmstash..."  << endl;
                }
                auto it = mmstash.find(make_pair(position_map[idx], idx));
                mmstash[make_pair(position_map[idx], idx)] = b;
            } else{
                // free dummy blocks to avoid memory leaks
                delete b;
            }
        }
        
    }
    
    // Read request data from stash
    for(int i = 0; i < ops.size(); i++){
        int blockIndex = blockIndices[i];
        int oldLeaf = position_map[blockIndex];
        int newLeaf = rand_gen->getRandomLeafWithBound(num_leaves);
        position_map[blockIndex] = newLeaf;
        Operation op = ops[i];
        int *data = new int[block_size];

        // cout << "Read index: " << blockIndex << " oldleaf: " << oldLeaf << " newLeaf: " << newLeaf << endl;

        auto it = mmstash.find(make_pair(oldLeaf, blockIndex));
        if(it != mmstash.end()){

            // Only support READ for now
            assert(op == Operation::READ);
            for(int i = 0; i < block_size; i++){
                data[i] = it->second->data[i];
            }
            ret.push_back(data);

            Block* b_tmp = it->second;
            mmstash.erase(it);

            // Assign this block to a new position
            mmstash[make_pair(newLeaf, blockIndex)] = b_tmp;
        } else{
            // This shouldn't happen
            cout << "missing block id: " << blockIndex << endl;
            assert(0);
        }
    } 

    if(!lazy_eveiction){
        evict_and_write_back();   
    }

    return ret;
}

void OramRing::evict_and_write_back(){

    // find the # evicted paths
    int num_paths = (cnt_delayed_leaves / evict_rate);
    // cout << "evict - num_paths: " << num_paths << endl;

    if(num_paths == 0){
        return;
    }

    // generate paths in reverse-lexicographical order
    vector<int> leaves;
    for(int i = 0; i < num_paths; i++){
        int g = ReverseBits(this->G, num_levels - 1);
        this->G = this->G + 1;
        leaves.push_back(g);
    }

    // Generate positions
    vector<int> positions; 
    vector<int> offsets;
    vector<bool> valids;
    set<int> position_cache; // avoid double evict in case of path overlap
    for (int l = num_levels - 1; l >= num_cached_levels; l--) {
        for(int leaf : leaves){
            int pos = P(leaf, l);
            // cout << "Request pos: " << pos << endl;
            // This position was not read before
            if(position_cache.find(pos) == position_cache.end()){
                int z = 0;
                for(int j = 0; j < bucket_size; j++){
                    if(metadata[pos].block_ids[j] != -1 && metadata[pos].valids[j] == 1){
                        // valid non-empty block
                        positions.push_back(pos);
                        offsets.push_back(j);
                        valids.push_back(true);
                        z++;
                    }
                }

                // // here if a bucket has been touched c times, then we should only read Z - c right?
                // int remain_real_block_size = real_bucket_size - metadata[pos].count;

                for(int j = 0; j < bucket_size; j++){
                    if(z >= real_bucket_size){
                        break;
                    }
                    if(metadata[pos].block_ids[j] == -1 && metadata[pos].valids[j] == 1){
                        // valid empty block
                        positions.push_back(pos);
                        offsets.push_back(j); 
                        valids.push_back(true);
                        z++;
                    }
                }
                position_cache.insert(pos);
                // make sure every bucket we read the same amount of blocks
                assert(z == real_bucket_size);
            }
        }
    }

    // Fetch the paths from remote
    std::vector<Block*> blocks;
    storage->ReadBlockBatchAsBlockRing(positions, offsets, blocks, valids, false);

    // Insert each block into stash
    for(Block* b : blocks){
        int idx = b->index;
        if (idx != -1) {
            auto it = mmstash.find(make_pair(position_map[idx], idx));
            if (it == mmstash.end()){
                mmstash[make_pair(position_map[idx], idx)] = b;
            } else{
                assert(0);
            }
        } else{
            delete b;
        }
    }

    // Evict
    std::random_device rd;  
    std::mt19937 g(rd());
    std::map<int, vector<Block*>> evicted_storage;
    std::set<int> visited;
    for (int l = num_levels - 1; l >= num_cached_levels; l--) {
        for(int leaf :  leaves){
            int Pxl = P(leaf, l);
            if(visited.find(Pxl) == visited.end()){
                vector<Block*> bkt;
                int counter = 0;

                std::pair<int, int> leaf_range = Q(leaf, l);
                for (auto it = mmstash.upper_bound(make_pair(leaf_range.first, 0)), next_it = it; it != mmstash.end(); it = next_it) {
                    ++next_it;

                    if (counter >= real_bucket_size) {
                        break;
                    }

                    if(it->first.first >= leaf_range.second){
                        break;
                    }

                    if (Pxl == P(position_map[it->first.second], l)) {
                        bkt.push_back(it->second);
                        // cout << "Evict: " << it->first.second << " pos: " << Pxl << " offset: " << bkt.size() - 1 << endl;
                        // cout << "Evict: " << it->first.second << " newLeaf: " << l << endl;
                        mmstash.erase(it);
                        counter++;
                        
                    } else{
                        cout << "oldleaf: " << leaf << endl;
                        cout << "l: " << l << endl;
                        cout << "leaf_range left: " << leaf_range.first << endl;
                        cout << "leaf_range right: " << leaf_range.second << endl;
                        cout << "current leaf id: " << it->first.first << endl;
                        cout << "current index: " << it->first.second << endl;
                        cout << "actual leaf id: " << position_map[it->first.second] << endl;
                        assert(0);
                    }
                }


                while(counter < bucket_size){
                    Block* dummy = new Block(block_size);
                    bkt.push_back(dummy); //dummy block
                    counter++;
                }                    

                // Shuffle and update metadata
                std::shuffle(bkt.begin(), bkt.end(), g);

                metadata[Pxl].count = 0;
                metadata[Pxl].valids.set();
                for(int i = 0; i < bucket_size; i++){
                    metadata[Pxl].block_ids[i] = bkt[i]->index;
                }

                evicted_storage[Pxl] = bkt;
                
                visited.insert(Pxl);
            }
            
        }
    } 

    storage->WriteBucketBatchMapAsBlockRing(evicted_storage, bucket_size, false);

    cnt_delayed_leaves = 0;
    cnt_q ++;

    // if(mmstash.size() != 0){
    //     cout << "evict mmsatsh size: " << mmstash.size() << " / " << num_cached_bucket * real_bucket_size << " resuffles: " << cnt_early_reshuffle*1.0 / cnt_q<< endl;
    // }
    // cnt_early_reshuffle = 0;
}

std::pair<int, int> OramRing::Q(int leaf, int level){
    int l = num_levels - 1 - level;
    // cout << "Q l: " << l << endl;
    int left = leaf >> l;
    // cout << "Q left: " << left << endl;
    left = left << l;
    // cout << "Q left: " << left << endl;
    int right = left + (1 << l);
    // cout << "Q right: " << right << endl;
    return std::make_pair(left, right);
}

int OramRing::P(int leaf, int level) {
    /*
    * This function should be deterministic. 
    * INPUT: leaf in range 0 to num_leaves - 1, level in range 0 to num_levels - 1. 
    * OUTPUT: Returns the location in the storage of the bucket which is at the input level and leaf.
    */
    return (1<<level) - 1 + (leaf >> (this->num_levels - level - 1));
}


/*
The below functions are to access various parameters, as described by their names.
INPUT: No input
OUTPUT: Value of internal variables given in the name.
*/

int* OramRing::getPositionMap() {
    return this->position_map.data();
}

void OramRing::loadPositionMap(std::vector<int>& pmap) {
    this->position_map = std::vector<int>();
    for(int i = 0; i < pmap.size(); i++){
        this->position_map.push_back(pmap[i]);
    }
    // for(int i =0; i < this->num_blocks; i++){
    //     this->position_map[i] = pmap[i];
    // }
}

void OramRing::loadPositionMapFromFile(const char* fname) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int s;
    fread(&s, 1, sizeof(int), f);
    this->position_map = std::vector<int>(s);
    fread(this->position_map.data(), 1, s*sizeof(int), f);
    // for(int i = 0; i < pmap.size(); i++){
    //     this->position_map.push_back(pmap[i]);
    // }
}

void OramRing::loadMetaDataFromFile(const char* fname) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int s;
    fread(&s, 1, sizeof(int), f);
    vector<int> block_ids = std::vector<int>(s);
    fread(block_ids.data(), 1, s*sizeof(int), f);
    assert(s == num_buckets*bucket_size);
    for(int i = 0; i < num_buckets; i++){
        metadata[i].count = 0;
        metadata[i].valids.set();
        metadata[i].block_ids = vector<int>(
            block_ids.begin() + i * bucket_size,
            block_ids.begin() + (i+1) * bucket_size
        );

        // cout << "metadata for "<< i << " : ";
        // for(auto id : metadata[i].block_ids){
        //     cout << id << " ";
        // }
        // cout << endl;

        // cout << "metadata for "<< i << " : ";
        // for(auto id : metadata[i].block_ids){
        //     cout << id << " ";
        // }
        // cout << endl;
    }

}

vector<Block> OramRing::getStash() {
    vector<Block> tmp;
    for (const auto& pair : this->mmstash) {
        tmp.push_back(Block(*(pair.second))); // Store the value in the vector
    }
    return tmp;
}
    
int OramRing::getStashSize() {
    return (this->mmstash).size();
}
    
int OramRing::getNumLeaves() {
    return this->num_leaves;

}

int OramRing::getNumLevels() {
    return this->num_levels;

}

int OramRing::getNumBlocks() {
    return this->num_blocks;

}

int OramRing::getNumBuckets() {
    return this->num_buckets;

}

