//
//
//

#ifndef PORAM_REMOTESERVERRING_H
#define PORAM_REMOTESERVERRING_H
#include "Bucket.h"
#include "net_io_channel.h"
#include <cmath>
#include <sstream>
#include <map>


class RemoteServerRing  {
public:

    // IO
    NetIO* io;

    // Integrity
    bool integrity;
    // std::vector<uint8_t> root;

    bool in_memory;
    
    // For Ring ORAM, we consider each bucket a block
    // std::vector<SBucket*> buckets;

    unsigned char* data;
    unsigned char* hash;

    size_t capacity;
    size_t bucket_size;
    size_t block_size;

    // If NOT
    // char* buckets_fname;
    // int fd;
    // unsigned char* mmap_data; // mmap data & file size
    // size_t mmap_size;
    // unsigned char* mmap_bkts; 

    RemoteServerRing(NetIO* io, size_t capacity, size_t bucket_size, bool in_memory, bool integrity);

    void RunServer();
    // void sync_root();

    void load(const char* fname);

    // Integrity related
    size_t per_bucket_tree_height;
	size_t per_bucket_hashes;
    uint8_t* per_bucket_hash;
    void load_hash(const char* fname);
    void sync_hash();
    
    void RunServerInMemory();

    void send_hash(std::vector<int> &position, std::vector<int> &offset);
    // void RunServerInDisk();
    // void RunServerInDiskRing();

    // void fileMap();
    // void fileUnMap();

};


#endif //PORAM_REMOTESERVERRING_H
