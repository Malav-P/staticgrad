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
         const std::string start);

void setup_model(GPT2*& model, const bool pretrained);
void setup_tokenizer(Tokenizer*& tk);
void setup_datastream(DataStream*& ds, const size_t numtokens);
void setup_activations(Activation*& activations,
                       Node*& out,
                       Node*& in, 
                       const size_t B,
                       const size_t T,
                       GPT2*& model);

void setup(GPT2*& model,
           DataStream*& ds,
           Tokenizer*& tk,
           Activation*& activations,
           Node*& out,
           Node*& in,
           const size_t B,
           const size_t T,
           const bool pretrained = false);

void tear_down(GPT2*& model,
               DataStream*& ds,
               Tokenizer*& tk,
               Activation*& activations,
               Node*& out,
               Node*& in);

            
void train(const int max_batches);


#endif //INTERFACE_HPP