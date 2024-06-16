#include "../include/tokenizer.hpp"

Tokenizer::Tokenizer(const std::string& filename){
    
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Unable to open file" << std::endl;
    }

    int token_id = 0;
    while (file.peek() != EOF) {
        // Read the length of the next string
        unsigned char length;
        file.read((char*)&length, sizeof(length));

        // Read the string
        std::string token(length, '\0');
        file.read(&token[0], length);

        token_map[token_id] = token;
        token_id++;
    }

    file.close();

    std::cout << "vocab size: " << token_map.size() << std::endl;
    std::cout << "First token: " << token_map[0] << std::endl;
    std::cout << "Second token: " << token_map[1] << std::endl;
    std::cout << "Last token: " << token_map[token_map.size()-1] << std::endl;

}

std::string Tokenizer::decodeSequence(int* tokenIDs, int length) {
    if (tokenIDs == nullptr){
        throw std::invalid_argument("input array is null");
    }
    std::string result;
    for (int i = 0; i < length; i++) {
        int key = tokenIDs[i];
        if (key < 0 || key >= token_map.size()){
            throw std::out_of_range("key is out of range of vocab size");
        }
        result += token_map[key];
    }

    return result;
}