#include <iostream>
#include <random>
#include <chrono>

float mc_cpu(float S0, float r, float sigma, float T, float k, int64_t N) {

    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    float payoff = 0;

    for (int64_t i = 0; i < N; i++) {
        float Z = dist(gen);
        float ST = S0 * std::exp((r - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * Z);
        payoff += std::max(ST - k, 0.0f);
    }

    float value = std::exp(-r * T) * (payoff / static_cast<float>(N));

    return value;
}


int main() {

    int64_t N = 100000000; // number of simulations

    float S0 = 100.0;   // initial stock price
    float r  = 0.05;    // risk-free rate
    float sigma = 0.2;  // volatility
    float T = 1.0;      // time
    float k = 110.0;    // strike price
    
    auto start = std::chrono::steady_clock::now();
    float result = mc_cpu(S0, r, sigma, T, k, N);
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> double_duration = end - start;
    std::cout << "Monte Carlo CPU Result: " << result << std::endl;
    std::cout << "Time taken: " << double_duration.count() << " milliseconds\n";

    return 0;

}