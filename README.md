# Monte Carlo Simulator

This project implements a Monte Carlo simulation to estimate the price of a European call option, a method commonly used in finance. It simulates many possible future stock prices and calculates the average payoff to estimate the option’s value.

## Features

Two versions were implememnted:

- **CPU version:** runs the simulation sequentially on a traditional processor.  
- **GPU version:** runs the simulation in parallel using CUDA, with each thread generating independent stock price paths for faster computation.  

## Tech Stack / Dependencies

- **Language:** C++17
- **Platform:** Google Colab (NVIDIA T4 GPU)  
- **GPU Acceleration:** CUDA 13.1
- **Compiler:** g++ for CPU, nvcc for GPU
- **Libraries:** cuRAND (for GPU random number generation)

> Google Colab comes pre-installed with the CUDA environment, but if running locally or in your own IDE, you need to install the CUDA Toolkit.*

## Option Parameters

The simulation uses several variables to define the European call option. These can be modified in the source code to explore different scenarios:

- **`S0` (Initial Stock Price):** The starting price of the underlying stock. Default is `100.0`.  
- **`r` (Risk-Free Rate):** The annualized risk-free interest rate, used for discounting the option payoff. Default is `0.05` (5%).  
- **`sigma` (Volatility):** The standard deviation of the stock’s returns, representing how much the stock price can fluctuate. Default is `0.2` (20%).  
- **`T` (Time to Maturity):** The time until the option expires, in years. Default is `1.0`.  
- **`k` (Strike Price):** The price at which the option can be exercised. Default is `110.0`.  

> To test different market conditions or option scenarios, simply modify these values in the code before compiling and running the simulation.


## How to Compile

You can run each versions independently, but you can also run both at the same time to compare performance metrics with ``` main.cpp ```. The instructions below assume you are running the simulations in the **Google Colab environment** with a T4 GPU.

```bash
# Compile CPU version (mc_cpu.cpp)
g++ -std=c++17 mc_cpu.cpp -o mc_cpu

# Compile GPU version (mc_gpu.cu)
nvcc -std=c++17 -arch=sm_75 mc_gpu.cu kernels.cu -o mc_gpu -lcurand

# Compile main (main.cpp)
g++ -std=c++17 main.cpp -o main
```

## How to Run

By default, the simulation runs with 1,000 paths. You can provide a command-line argument to specify the number of simulations. Values up beyond 10<sup>8</sup> will take some time on the CPU version of the program, but the GPU version should complete in a reasonable amount of time. Use moderate simulation counts to avoid long runtimes or unexpected behavior.

```bash
# Run CPU simulation
./mc_cpu [num_simulations]

# Run GPU simulation
./mc_gpu [num_simulations]

# Run Both
./main [num_simulations]
```

## References 

- [Monte Carlo methods for option pricing](https://en.wikipedia.org/wiki/Monte_Carlo_methods_for_option_pricing)
- [CUDA Toolkit Documentation 13.1](https://docs.nvidia.com/cuda/index.html)
