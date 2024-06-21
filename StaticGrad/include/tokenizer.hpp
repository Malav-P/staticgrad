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

        std::string decode(u_int16_t* tokenIDs, int length);
        std::string decode(const std::vector<u_int16_t>& tokenIDs);

        std::vector<u_int16_t> encode(const std::string& str);
};

#endif // TOKENIZER_HPP