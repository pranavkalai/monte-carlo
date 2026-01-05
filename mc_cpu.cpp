// %%writefile mc_cpu.cpp
#include <iostream>
#include <random>
#include <chrono>

double mc_cpu(double S0, double r, double sigma, double T, double k, int64_t N) {

    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    double payoff = 0;

    for (int64_t i = 0; i < N; i++) {
        double Z = dist(gen);
        double ST = S0 * std::exp((r - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * Z);
        payoff += std::max(ST - k, 0.0);
    }

    double value = std::exp(-r * T) * (payoff / static_cast<double>(N));

    return value;
}


int main(int argc, char* argv[]) {

    int64_t N = 1000; // number of simulations, can be changed by providing a cmd line argument
    if (argc == 2) {
        N = std::stoll(argv[1]);
    }
    if (argc > 2) {
        std::cout << "Too many arguements provided" << std::endl;
        return -1;
    }

    double S0 = 100.0;   // initial stock price
    double r  = 0.05;    // risk-free rate
    double sigma = 0.2;  // volatility
    double T = 1.0;      // time
    double k = 110.0;    // strike price

    auto start = std::chrono::steady_clock::now();
    double result = mc_cpu(S0, r, sigma, T, k, N);
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> double_duration = end - start;
    std::cout << "Monte Carlo CPU Result: " << result << std::endl;
    std::cout << "Time taken: " << double_duration.count() << " milliseconds\n";

    return 0;

}