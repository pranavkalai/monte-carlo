#include <cuda_runtime.h>
#include <kernels.h>
#include <iostream>

int main() {

    int64_t N = pow(10, 9); // number of simulations

    float S0 = 100.0;   // initial stock price
    float r  = 0.05;    // risk-free rate
    float sigma = 0.2;  // volatility
    float T = 1.0;      // time
    float k = 110.0;    // strike price

    int threads_per_block = 256;
    int blocks_per_grid = 1024;
    int num_threads = threads_per_block * blocks_per_grid;

    curandStatePhilox4_32_10_t* d_states;
    cudaMalloc(&d_states, num_threads * sizeof(curandStatePhilox4_32_10_t));

    init_kernel<<<blocks_per_grid, threads_per_block>>>(d_states, 42);

    float* results_d;
    cudaMalloc(&results_d, num_threads * sizeof(float));

    monte_carlo_kernel<<<blocks_per_grid, threads_per_block>>>(d_states, S0, r, sigma, T, k, N, results_d, num_threads);

    cudaDeviceSynchronize(); // wait for gpu stuff to finish

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


