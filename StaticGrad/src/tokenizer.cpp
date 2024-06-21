#include "../include/tokenizer.hpp"

 /**
  * Constructs a `Tokenizer` object from a binary file.
  *
  * @param filename Path to the binary file containing token data.
  * 
  * @note - The file format is expected to be a bytestream of token_length-token-token_length-token...
  * @note - The constructor reads the file, populates the `token_map` and prints statistics on the vocabulary size and the first, second, and last tokens.
  */
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

    // // for debugging, can remove
    // std::cout << "vocab size: " << token_map.size() << std::endl;
    // std::cout << "First token: " << token_map[0] << std::endl;
    // std::cout << "Second token: " << token_map[1] << std::endl;
    // std::cout << "Last token: " << token_map[token_map.size()-1] << std::endl;

}

/**
 * Decode the given array of token IDs into a human-readable string.
 *
 * @param tokenIDs Pointer to an array of token IDs.
 * @param length The number of elements in the `tokenIDs` array.
 * 
 * @return The decoded string.
 * 
 * @throw `std::invalid_argument` If the `tokenIDs` pointer is null.
 * @throw `std::out_of_range` If any token ID in the array is invalid (out of the range of the vocabulary).
 */ 
std::string Tokenizer::decode(u_int16_t* tokenIDs, int length) {
    if (tokenIDs == nullptr){
        throw std::invalid_argument("input array is null");
    }
    std::string result;
    for (int i = 0; i < length; i++) {
        u_int16_t key = tokenIDs[i];
        if (key < 0 || key >= token_map.size()){
            throw std::out_of_range("key is out of range of vocab size");
        }
        result += token_map[key];
    }

    return result;
}

/**
* Overloaded function to decode a `std::vector` of token IDs into a human-readable string.
*
* @param tokenIDs A vector of token IDs.
*
* @return The decoded string.
*
* @throw `std::out_of_range` If any token ID in the vector is invalid.
*/ 
std::string Tokenizer::decode(const std::vector<u_int16_t>& tokenIDs) {

    std::string result;
    for (int i = 0; i < tokenIDs.size(); i++) {
        u_int16_t key = tokenIDs[i];
        if (key < 0 || key >= token_map.size()){
            throw std::out_of_range("key is out of range of vocab size");
        }
        result += token_map[key];
    }

    return result;
}

/**
* Encode the given string into a sequence of token IDs.
*
* @param str The input string to be tokenized.
*
* @return A vector of token IDs representing the input string.
*
* @throw `std::runtime_error` If there are unknown tokens in the input string.
*/
std::vector<u_int16_t> Tokenizer::encode(const std::string& str) {
    std::vector<u_int16_t> tokenIDs;
    std::string remainingStr = str;

    while (!remainingStr.empty()) {
        bool found = false;
        int maxLength = 0;
        u_int16_t maxTokenId = 0;

        for (const auto& pair : token_map) {
            if (remainingStr.find(pair.second) == 0) {
                if (pair.second.size() > maxLength) {
                    maxLength = pair.second.size();
                    maxTokenId = pair.first;
                    found = true;
                }
            }
        }

        if (found) {
            tokenIDs.push_back(maxTokenId);
            remainingStr.erase(0, maxLength);
        } else {
            throw std::runtime_error("Unknown token in input string");
        }
    }

    return tokenIDs;
}