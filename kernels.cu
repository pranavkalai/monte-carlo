// %%writefile kernels.cu
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "kernels.h"

__global__ void init_kernel(curandStatePhilox4_32_10_t* device_states_, unsigned long long seed, int num_threads_) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // thread index calculation
    if (idx >= num_threads_) {return;}
    curand_init(seed, idx, 0, &device_states_[idx]); // initialize ecah thread's CURAND state

}

__global__ void monte_carlo_kernel(curandStatePhilox4_32_10_t* device_states_,
    float S0, float r, float sigma, float T, float k, int64_t N, float* results, int num_threads_) {

    float payoff = 0;

    int thrd_idx = blockIdx.x * blockDim.x + threadIdx.x; // thread index calculation
    if (thrd_idx >= num_threads_) {return;}
    int stride = blockDim.x * gridDim.x; // total number of threads

    curandStatePhilox4_32_10_t local_state = device_states_[thrd_idx]; // copy state to local memory

    for (int64_t i = thrd_idx; i < N; i += stride) {
        float normal_rand_var = curand_normal(&local_state); // generate normal random variable

        // explore optimizing exp and sqrt calls
        float ST = S0 * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * normal_rand_var);

        payoff += fmax(ST - k, 0.0f); // partial sum of payoffs

        //if (thrd_idx < 4 && i < 10) { // debug statements
           // printf("tid=%d, i=%lld, payoff=%f\n", thrd_idx, i, payoff);
        //}
    }

    results[thrd_idx] = payoff; // store result
    device_states_[thrd_idx] = local_state; // update state in global memory

}