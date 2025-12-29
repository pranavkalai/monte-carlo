#include <curand_kernel.h>

__global__ void init(curandStatePhilox4_32_10_t* device_states_, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // thread index calculation
    curand_init(seed, idx, 0, &device_states_[idx]); // initialize ecah thread's CURAND state
}

__global__ void monte_carlo_kernel(curandStatePhilox4_32_10_t* device_states_,
    float S0, float r, float sigma, float T, float k, int64_t N, float* payoff) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // thread index calculation

    if (idx >= N) {return;} // boundary check

    curandStatePhilox4_32_10_t local_state = device_states_[idx]; // copy state to local memory
    float normal_rand_var = curand_normal(&local_state); // generate normal random variable

    // explore optimizing exp and sqrt calls
    float ST = S0 * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * normal_rand_var);

    *payoff += fmax(ST - k, 0.0f); // calculate payoff for this path

}