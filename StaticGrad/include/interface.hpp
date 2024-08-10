#ifndef INTERFACE_HPP
#define INTERFACE_HPP

#include <cstddef>
#include <string>


class GPT2;
class DataStream;
class Tokenizer;
class Activation;
class Node;

void yap(GPT2*& model,
         Tokenizer*& tk,
         Node*& out,
         Node*& in,
         std::string start);


void setup(GPT2*& model,
           DataStream*& ds,
           Tokenizer*& tk,
           Activation*& activations,
           Node*& out,
           Node*& in,
           size_t B,
           size_t T,
           bool pretrained = false);

void tear_down(GPT2*& model,
               DataStream*& ds,
               Tokenizer*& tk,
               Activation*& activations,
               Node*& out,
               Node*& in);

            
void train(int max_batches);


#endif //INTERFACE_HPP