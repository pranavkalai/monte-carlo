#include <curand_kernel.h>

#pragma once

extern __global__ void init(curandStatePhilox4_32_10_t* device_states_, unsigned long long seed);