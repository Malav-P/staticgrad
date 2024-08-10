#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <string>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>


class Tokenizer {
    public:

        std::unordered_map<int, std::string> token_map;

        Tokenizer(const std::string& filename);

        std::string decode(const uint16_t* tokenIDs, const int length);
        std::string decode(const std::vector<uint16_t>& tokenIDs);

        std::vector<uint16_t> encode(const std::string& str);
};

#endif // TOKENIZER_HPP