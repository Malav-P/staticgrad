#ifndef INTERFACE_HPP
#define INTERFACE_HPP

#include <cstddef>

class GPT2;
class DataStream;
class Tokenizer;
class Node;

void setup(GPT2*& model,
           DataStream*& datastream,
           Tokenizer*& tk,
           Node*& out, 
           Node*& in,
           size_t B,
           size_t T,
           bool pretrained = false);

void tear_down( GPT2*& model,
                DataStream*& ds,
                Tokenizer*& tk,
                Node*& out, 
                Node*& in);

            
void train(int max_batches);

#endif //INTERFACE_HPP