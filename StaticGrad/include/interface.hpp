#ifndef INTERFACE_HPP
#define INTERFACE_HPP

#include <cstddef>

class GPT2;
class DataStream;
class Tokenizer;
class Node;

void allocate_activations(Node*& out,
                          Node*& in,
                          size_t B,
                          size_t T,
                          size_t C,
                          size_t L,
                          size_t V);

void deallocate_activations(Node*& out,
                            Node*& in);

void setup(GPT2*& model,
           DataStream*& datastream,
           Tokenizer*& tk,
           bool pretrained = false);

void tear_down( GPT2*& model,
                DataStream*& ds,
                Tokenizer*& tk);

            
void train(int max_batches);


#endif //INTERFACE_HPP