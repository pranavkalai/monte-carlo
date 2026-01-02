// %%writefile mc_gpu.cu
#include <cuda_runtime.h>
#include <iostream>
#include "kernels.h"

int main() {

    int64_t N = 1000000000; // number of simulations

    float S0 = 100.0;   // initial stock price
    float r  = 0.05;    // risk-free rate
    float sigma = 0.2;  // volatility
    float T = 1.0;      // time
    float k = 110.0;    // strike price

    int threads_per_block = 256;
    int blocks_per_grid = 256;
    int num_threads = threads_per_block * blocks_per_grid;

    curandStatePhilox4_32_10_t* d_states;
    cudaMalloc(&d_states, num_threads * sizeof(curandStatePhilox4_32_10_t));

    init_kernel<<<blocks_per_grid, threads_per_block>>>(d_states, 42, num_threads);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("init_kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    }

    float* results_d;
    cudaMalloc(&results_d, num_threads * sizeof(float));

    monte_carlo_kernel<<<blocks_per_grid, threads_per_block>>>(d_states, S0, r, sigma, T, k, N, results_d, num_threads);

    cudaError_t cudaerr1 = cudaDeviceSynchronize();
    if (cudaerr1 != cudaSuccess) {
        printf("init_kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr1));
    }

    float* results_h = new float[num_threads];
    cudaMemcpy(results_h, results_d, num_threads * sizeof(float), cudaMemcpyDeviceToHost);

    float sum_payoff = 0;
    for (int64_t i = 0; i < num_threads; i++) {
        sum_payoff += results_h[i];
    }

    float value = std::exp(-r * T) * (sum_payoff / static_cast<float>(N));

    std::cout << "Monte Carlo GPU Result: " << value << std::endl;


    return 0;
}