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
           bool pretrained = false);

void tear_down( GPT2*& model,
                DataStream*& ds,
                Tokenizer*& tk);

            
void train(int max_batches);


#endif //INTERFACE_HPP