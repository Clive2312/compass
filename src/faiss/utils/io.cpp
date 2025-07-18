#include "io.h"

float* pca_read(const char* fname, size_t* d_out, size_t* n_out, std::vector<int> &id_list) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    fread(&d, 1, sizeof(int), f);

    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;

    // printf("file size: %ld\n", sz);
    assert(sz % ((d + 2) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 2) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 2)];
    size_t nr = fread(x, sizeof(float), n * (d + 2), f);
    assert(nr == n * (d + 2) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++){
        id_list.push_back(((int*)x)[i * (d + 2)]);
        memmove(x + i * d, x + 2 + i * (d + 2), d * sizeof(*x));
    }

    fclose(f);
    return x;
}

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