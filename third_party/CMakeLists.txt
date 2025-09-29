find_package(OpenSSL REQUIRED)

add_library(rapidjson INTERFACE)
target_include_directories(rapidjson INTERFACE rapidjson/include/)

add_library(argmap INTERFACE)
target_include_directories(argmap INTERFACE argmap/)

add_library(net INTERFACE)
target_include_directories(net INTERFACE net/)

add_library(evp openssl/evp.h openssl/evp.cpp)
target_include_directories(evp PUBLIC openssl/)
target_link_libraries(evp PRIVATE OpenSSL::SSL OpenSSL::Crypto)

add_subdirectory(SEAL)

# Build FAISS from submodule
set(FAISS_ENABLE_GPU OFF CACHE BOOL "Disable GPU support" FORCE)
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "Disable Python support" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable tests" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static library" FORCE)
add_subdirectory(faiss)