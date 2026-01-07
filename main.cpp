// %%writefile main.cpp
#include <iostream>
#include <unistd.h>
#include <sys/wait.h>

int main(int argc, char* argv[]) {

    char* args_cpu[] = {(char*)"./mc_cpu", (char*)"1000", nullptr};
    char* args_gpu[] = {(char*)"./mc_gpu", (char*)"1000", nullptr};

    if (argc == 2) {
        args_cpu[1] = argv[1];
        args_gpu[1] = argv[1];
    }
    else if (argc > 2) {
        std::cerr << "Too many arguements provided" << std::endl;
        return -1;
    }

    std::cout << "Running CPU and GPU Monte Carlo Simulations with N = " << args_cpu[1] << std::endl;
    std::cout << std::endl;

    pid_t pid = fork();

    if (pid < 0) {
        std::cerr << "Fork failed" << std::endl;
        return -1;
    } 
    else if (pid == 0) {
        // Child process
        if (execvp(args_cpu[0], args_cpu) < 0) {
            std::cerr << "Error executing CPU program" << std::endl;
            return -1;
        }
    } 
    else {
        // Parent process
        waitpid(pid, nullptr, 0); // Wait for child process to finish
        std::cout << std::endl;
        if (execvp(args_gpu[0], args_gpu) < 0) {
            std::cerr << "Error executing GPU program" << std::endl;
            return -1;
        }
    }

    return 0;
    
}