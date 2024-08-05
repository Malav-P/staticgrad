#include "interface.hpp"
#include <iostream>

int main(int argc, char **argv) {

    int iter = std::stoi(argv[1]);
    train(iter);

    return 0;
}