#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <string>
#include <fstream>
#include <iostream>
#include <unordered_map>


class Tokenizer {
    public:

        std::unordered_map<int, std::string> token_map;

        Tokenizer(const std::string& filename);

        std::string decodeSequence(int* tokenIDs, int length);
};

#endif // TOKENIZER_HPP