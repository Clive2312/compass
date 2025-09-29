set(FAISS_SRC
    impl/AuxIndexStructures.cpp
    impl/FaissException.cpp
    impl/HNSW.cpp
    impl/oblivious_hnsw.cpp
    impl/io.cpp
    impl/ProductQuantizer.cpp
    utils/Heap.cpp
    utils/random.cpp
    utils/io.cpp
    IndexHNSW.cpp
    index_io.cpp
)

# set(FAISS_HEADERS
#     impl/AuxIndexStructures.h
#     impl/DistanceComputer.h
#     impl/FaissAssert.h
#     impl/FaissException.h
#     utils/Heap.h
#     utils/ordered_key_value.h
#     utils/prefetch.h
#     utils/random.h
#     utils/io.h
# )

add_library(compass_faiss ${FAISS_SRC})
target_link_libraries(compass_faiss PRIVATE oram_library)

# set(FAISS_HEADERS ${FAISS_HEADERS} PARENT_SCOPE)