#include <curand_kernel.h>

#pragma once

extern __global__ void init(curandStatePhilox4_32_10_t* device_states_, unsigned long long seed);

extern __global__ void monte_carlo_kernel(curandStatePhilox4_32_10_t* device_states_,
    float S0, float r, float sigma, float T, float k, int64_t N, float* payoff);