#include <cuda_runtime.h>
#include <kernels.h>
#include <iostream>

int main() {

    curandStatePhilox4_32_10_t* d_states;
    cudaMalloc(&d_states, 1024 * sizeof(curandStatePhilox4_32_10_t));

    // init<<<1, 1024>>>(d_states, 42ULL);

    float* payoff;
    cudaMalloc(&payoff, sizeof(float));
    cudaMemset(payoff, 0, sizeof(float));

    
    return 0;
}


