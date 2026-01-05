// %%writefile mc_gpu.cu
#include <cuda_runtime.h>
#include <iostream>
#include "kernels.h"

int main(int argc, char* argv[]) {

    int64_t N = 1000; // number of simulations, can be changed by providing a cmd line argument
    if (argc == 2) {
        N = std::stoll(argv[1]);
    }
    if (argc > 2) {
        std::cout << "Too many arguements provided" << std::endl;
        return -1;
    }

    // Variables for the option pricing, these can be changed
    float S0 = 100.0;   // initial stock price
    float r  = 0.05;    // risk-free rate
    float sigma = 0.2;  // volatility
    float T = 1.0;      // time
    float k = 110.0;    // strike price

    // 256 by 256 seems to be the most optimal setup for Google Colab's GPU (T4), could experimnent more with this
    int threads_per_block = 256;
    int blocks_per_grid = 256;
    int num_threads = threads_per_block * blocks_per_grid;

    // Time mesurement events
    cudaEvent_t start_init, start_mc, stop_init, stop_mc;
    cudaEventCreate(&start_init);
    cudaEventCreate(&stop_init);
    cudaEventCreate(&start_mc);
    cudaEventCreate(&stop_mc);

    // Philox4_32_10 is one of the RNG algorithms provided by NVIDA's cuRAND library, could experiment with others
    curandStatePhilox4_32_10_t* d_states;
    cudaMalloc(&d_states, num_threads * sizeof(curandStatePhilox4_32_10_t)); // allocate space for an array of RNG states

    cudaEventRecord(start_init); // This just records a timestamp, to measure the kernel's execution time
    init_kernel<<<blocks_per_grid, threads_per_block>>>(d_states, 42, num_threads); // initialize RNG states
    cudaEventRecord(stop_init);
    
    // error checking
    cudaError_t cudaerr = cudaEventSynchronize(stop_init);
    if (cudaerr != cudaSuccess) {
        printf("init_kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    }

    float* results_d;
    cudaMalloc(&results_d, num_threads * sizeof(float)); // allocates space for the partial sums of 
                                                         // the payoffs computed by each thread

    cudaEventRecord(start_mc);
    monte_carlo_kernel<<<blocks_per_grid, threads_per_block>>>(d_states, S0, r, sigma, T, k, N, results_d, num_threads);
    cudaEventRecord(stop_mc);


    cudaError_t cudaerr1 = cudaEventSynchronize(stop_mc);
    if (cudaerr1 != cudaSuccess) {
        printf("init_kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr1));
    }

    float* results_h = new float[num_threads];
    cudaMemcpy(results_h, results_d, num_threads * sizeof(float), cudaMemcpyDeviceToHost); // copies the results array from the GPU RAM
                                                                                           // to the CPU RAM
    float sum_payoff = 0;
    for (int64_t i = 0; i < num_threads; i++) {
        sum_payoff += results_h[i]; // summing up the partial payoffs
    }

    float value = std::exp(-r * T) * (sum_payoff / static_cast<float>(N));

    // compute the execution times
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