#include "interface.hpp"
#include <iostream>

int main(int argc, char **argv) {

    int iter = 5;
    if (argc == 2){
        iter = std::stoi(argv[1]);
    }
 
    train(iter);

    return 0;
}