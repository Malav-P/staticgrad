#ifndef INFERENCE_HPP
#define INFERENCE_HPP

class GPT2;
class Tokenizer;
class Node;

#include <string>

uint16_t next_token(GPT2*& model,
                     Node*& out,
                     Node*& in,
                     size_t t);


void yap(GPT2*& model,
         Tokenizer*& tk,
         Node*& out,
         Node*& in,
         std::string start);



#endif // INFERENCE_HPP