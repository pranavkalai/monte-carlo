#include <cuda_runtime.h>
#include <iostream>
#include <kernels.h>

int main() {

    curandStatePhilox4_32_10_t* d_states;
    cudaMalloc(&d_states, 1024 * sizeof(curandStatePhilox4_32_10_t));

    init<<<1, 1024>>>(d_states, 42ULL);

    
    return 0;
}


