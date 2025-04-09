#ifndef INTERFACE_HPP
#define INTERFACE_HPP

#include <cstddef>
#include <string>


class GPT2;
class DataStream;
class Tokenizer;
class Activation;
class Node;
class Optimizer;

enum optimizer_t {
    ADAM
};

size_t prepare_for_gen(GPT2*& model,
                       Tokenizer*& tk,
                       Node*& in,
                       std::string start);
std::string next_token(GPT2*& model,
                     Tokenizer*& tk,
                     Node*& out,
                     Node*& in,
                     const size_t t);
void yap(GPT2*& model,
         Tokenizer*& tk,
         Node*& out,
         Node*& in,
         const std::string start);

void setup_model(GPT2*& model, const bool pretrained);
void setup_optimizer(Optimizer*& opt, GPT2*& model, optimizer_t opt_name);
void setup_tokenizer(Tokenizer*& tk);
void setup_datastream(DataStream*& ds, const size_t numtokens);
void setup_activations(Activation*& activations,
                       Node*& out,
                       Node*& in, 
                       const size_t B,
                       const size_t T,
                       GPT2*& model);

void setup(GPT2*& model,
           Optimizer*& opt,
           DataStream*& ds,
           Tokenizer*& tk,
           Activation*& activations,
           Node*& out,
           Node*& in,
           const size_t B,
           const size_t T,
           const bool pretrained = false);

void teardown_model(GPT2*& model);
void teardown_optimizer(Optimizer*& opt);
void teardown_tokenizer(Tokenizer*& tk);
void teardown_datastream(DataStream*& ds);
void teardown_activations(Activation*& activations, Node*& out, Node*& in);

void tear_down(GPT2*& model,
               Optimizer*& opt,
               DataStream*& ds,
               Tokenizer*& tk,
               Activation*& activations,
               Node*& out,
               Node*& in);

// clear kv cache and reset activations (the latter is hypothesized to help cache timing as it speeds up inference)
void clear_cache(GPT2*& model, Activation*& activations);

            
void train(const int max_batches);


#endif //INTERFACE_HPP