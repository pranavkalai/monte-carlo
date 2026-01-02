// %%writefile mc_gpu.cu
#include <cuda_runtime.h>
#include <iostream>
#include "kernels.h"

int main(int argc, char* argv[]) {

    int64_t N = 1000; // number of simulations
    if (argc == 2) {
        N = std::stoll(argv[1]);
    }
    if (argc > 2) {
        std::cout << "Too many arguements provided" << std::endl;
        return -1;
    }

    float S0 = 100.0;   // initial stock price
    float r  = 0.05;    // risk-free rate
    float sigma = 0.2;  // volatility
    float T = 1.0;      // time
    float k = 110.0;    // strike price

    int threads_per_block = 256;
    int blocks_per_grid = 256;
    int num_threads = threads_per_block * blocks_per_grid;

    // Time mesurement events
    cudaEvent_t start_init, start_mc, stop_init, stop_mc;
    cudaEventCreate(&start_init);
    cudaEventCreate(&stop_init);
    cudaEventCreate(&start_mc);
    cudaEventCreate(&stop_mc);

    curandStatePhilox4_32_10_t* d_states;
    cudaMalloc(&d_states, num_threads * sizeof(curandStatePhilox4_32_10_t));

    cudaEventRecord(start_init);
    init_kernel<<<blocks_per_grid, threads_per_block>>>(d_states, 42, num_threads);
    cudaEventRecord(stop_init);
    

    cudaError_t cudaerr = cudaEventSynchronize(stop_init);
    if (cudaerr != cudaSuccess) {
        printf("init_kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    }

    float* results_d;
    cudaMalloc(&results_d, num_threads * sizeof(float));

    cudaEventRecord(start_mc);
    monte_carlo_kernel<<<blocks_per_grid, threads_per_block>>>(d_states, S0, r, sigma, T, k, N, results_d, num_threads);
    cudaEventRecord(stop_mc);


    cudaError_t cudaerr1 = cudaEventSynchronize(stop_mc);
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

    float time_init, time_mc;
    cudaEventElapsedTime(&time_init, start_init, stop_init);
    cudaEventElapsedTime(&time_mc, start_mc, stop_mc);

    std::cout << "Monte Carlo GPU Result: " << value << std::endl;
    std::cout << "Time taken (init_kernel): " << time_init << " milliseconds\n";
    std::cout << "Time taken (monte_carlo_kernel): " << time_mc << " milliseconds\n";
    std::cout << "Total Time taken: " << time_init + time_mc << " milliseconds\n";

    cudaEventDestroy(start_init);
    cudaEventDestroy(stop_init);
    cudaEventDestroy(start_mc);
    cudaEventDestroy(stop_mc);

    cudaFree(d_states);
    cudaFree(results_d);
    delete[] results_h;

    return 0;
}