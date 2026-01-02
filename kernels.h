// %%writefile kernels.h
#include <curand_kernel.h>

#pragma once

extern __global__ void init_kernel(curandStatePhilox4_32_10_t* device_states_, unsigned long long seed, int num_threads_);

extern __global__ void monte_carlo_kernel(curandStatePhilox4_32_10_t* device_states_,
    float S0, float r, float sigma, float T, float k, int64_t N, float* results, int num_threads_);