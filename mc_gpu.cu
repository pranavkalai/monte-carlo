// %%writefile mc_gpu.cu
#include <cuda_runtime.h>
#include <iostream>
#include <curand_kernel.h>
#include "kernels.h"

int main() {

    int64_t N = 100000000; // number of simulations

    float S0 = 100.0;   // initial stock price
    float r  = 0.05;    // risk-free rate
    float sigma = 0.2;  // volatility
    float T = 1.0;      // time
    float k = 110.0;    // strike price

    int threads_per_block = 256;
    int blocks_per_grid = 1;
    int num_threads = threads_per_block * blocks_per_grid;

    curandStatePhilox4_32_10_t* d_states;
    cudaMalloc(&d_states, num_threads * sizeof(curandStatePhilox4_32_10_t));

    cudaEvent_t start_init, stop_init;
    cudaEventCreate(&start_init);
    cudaEventCreate(&stop_init);

    cudaEventRecord(start_init);
    init_kernel<<<blocks_per_grid, threads_per_block>>>(d_states, 42);
    cudaEventRecord(stop_init);
    cudaEventSynchronize(stop_init);

    float init_ms = 0;
    cudaEventElapsedTime(&init_ms, start_init, stop_init);
    std::cout << "init_kernel time: " << init_ms << " ms\n";

    float* results_d;
    cudaMalloc(&results_d, num_threads * sizeof(float));

    cudaEvent_t start_mc, stop_mc;
    cudaEventCreate(&start_mc);
    cudaEventCreate(&stop_mc);

    cudaEventRecord(start_mc);
    monte_carlo_kernel<<<blocks_per_grid, threads_per_block>>>(d_states, S0, r, sigma, T, k, N, results_d, num_threads);
    cudaEventRecord(stop_mc);
    cudaEventSynchronize(stop_mc);

    float mc_ms = 0;
    cudaEventElapsedTime(&mc_ms, start_mc, stop_mc);
    std::cout << "monte_carlo_kernel time: " << mc_ms << " ms\n";

    float total_ms = init_ms + mc_ms;
    std::cout << "Total GPU kernel time: " << total_ms << " ms\n";

    float* results_h = new float[num_threads];
    cudaMemcpy(results_h, results_d, num_threads * sizeof(float), cudaMemcpyDeviceToHost);

    float sum_payoff = 0;
    for (int64_t i = 0; i < num_threads; i++) {
        sum_payoff += results_h[i];
    }

    float value = std::exp(-r * T) * (sum_payoff / static_cast<float>(N));
    std::cout << "Monte Carlo GPU Result: " << value << std::endl;

    cudaFree(d_states);
    cudaFree(results_d);
    delete[] results_h;

    cudaEventDestroy(start_init);
    cudaEventDestroy(stop_init);
    cudaEventDestroy(start_mc);
    cudaEventDestroy(stop_mc);

    return 0;
}
