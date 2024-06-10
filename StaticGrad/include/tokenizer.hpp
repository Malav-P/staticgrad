#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <string>
#include <fstream>
#include <iostream>
#include <unordered_map>


class Tokenizer {
    public:

        std::unordered_map<int, std::string> token_map;

        Tokenizer(std::string& filename);
};

#endif // TOKENIZER_HPP