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

*Note: Google Colab comes pre-installed with the CUDA environment, but if running locally or in your own IDE, you need to install the CUDA Toolkit.*

## How to Compile

You can run each versions independently, but you can also run both at the same time to compare performance metrics with ``` main.cpp ```

```bash
# Compile CPU version (mc_cpu.cpp)
g++ -std=c++17 mc_cpu.cpp -o mc_cpu

# Compile GPU version (mc_gpu.cu)
nvcc -std=c++17 -arch=sm_75 mc_gpu.cu kernels.cu -o mc_gpu -lcurand

# Compile main (main.cpp)
g++ -std=c++17 main.cpp -o main
```

## How to Run

By default, the simulation runs with 1,000 paths. You can provide a command-line argument to specify the number of simulations. Values up to 10<sup>8</sup> have been tested; beyond that, the GPU version should still work, but the CPU version may take a very long time. Bugs are still being fixed, so use moderate simulation counts to avoid long runtimes or unexpected behavior.

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
