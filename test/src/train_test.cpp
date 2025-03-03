#include "interface.hpp"
#include <iostream>
#include <cstdlib> // for std::exit
#include <string> 

int main(int argc, char** argv) {
    int iter = 5; // Default value

    if (argc == 2) {
        try {
            iter = std::stoi(argv[1]);
            if (iter <= 0) {
                std::cerr << "Error: The number of training epochs must be a positive integer." << std::endl;
                return 1; // Non-zero return indicates an error
            }
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error: Invalid argument. Please pass a positive integer." << std::endl;
            return 1; // Non-zero return indicates an error
        } catch (const std::out_of_range& e) {
            std::cerr << "Error: Number out of range. Please pass a valid positive integer." << std::endl;
            return 1; // Non-zero return indicates an error
        }
    }

    train(iter);
    return 0; // Indicate success
}