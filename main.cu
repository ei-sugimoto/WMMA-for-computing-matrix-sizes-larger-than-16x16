#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

typedef signed char  ElementAB;
typedef int ElementC;

__global__ void wmma_ker(const ElementAB* a, const ElementAB* b, ElementC* c, const int N) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, ElementAB, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, ElementAB, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, ElementC> c_frag;

       
    // Initialize the output to zero
    wmma::fill_fragment(c_frag, __float2half(.0f));
    // Load the inputs

    for (auto k = 0; k < N; k += 16) {
        wmma::load_matrix_sync(a_frag, &a[blockIdx.y * N * 16 + k], N);
        wmma::load_matrix_sync(b_frag, &b[k * N + blockIdx.x * 16], N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
   // Store the output
    wmma::store_matrix_sync(&c[blockIdx.y * N * 16 + blockIdx.x * 16], c_frag, N, wmma::mem_row_major);
}

int main() {

    ElementAB *inputA, *inputB;
    ElementC *outputC;
    const int N = 16;
    const int size = N * N;
    cudaMallocManaged(&inputA, size * sizeof(ElementAB));
    cudaMallocManaged(&inputB, size * sizeof(ElementAB));
    cudaMallocManaged(&outputC, size * sizeof(ElementC));

    for (int i = 0; i < size; i++) {
        inputA[i] = 1;
        inputB[i] = 1;
    }
    
    wmma_ker<<<dim3(N / 16, N / 16), dim3(16, 16)>>>(inputA, inputB, outputC, N);

    cudaDeviceSynchronize();

    for (int i = 0; i < size; i++) {
        std::cout << outputC[i];
    }
    std::cout << std::endl;

    cudaFree(inputA);
    cudaFree(inputB);
    cudaFree(outputC);

    return 0;
}
