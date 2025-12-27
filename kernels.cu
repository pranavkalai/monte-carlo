#include <curand_kernel.h>

__global__ void init(curandStatePhilox4_32_10_t* device_states_, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // thread index calculation
    curand_init(seed, idx, 0, &device_states_[idx]); // initialize ecah thread's CURAND state
}