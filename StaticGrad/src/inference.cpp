#include "tokenizer.hpp"
#include "gpt2.hpp"
#include "inference.hpp"
#include "utils.hpp"

/**
 * Computes the most probable token at specified position
 *
 * Args:
 *   @param model: Pointer to instance of GPT2
 *   @param out: The output node where the computed loss will be stored, a 2D tensor of shape (B, T).
 *   @param in: The input node containing the probabilities, a 3D tensor of shape (B, T, V).
 *   @param t: Desired position of next token in sequence
 *
 * Returns:
 *   uint16_t. The most probable token for that position
 */
uint16_t next_token(GPT2*& model,
                     Node*& out,
                     Node*& in,
                     size_t t){

    if (t == 0){
        throw std::runtime_error("t cannot be zero");
    }

    size_t B = out->shape[0];
    size_t V = out->shape[2];

    in->current_T = t;

    if (B != 1){
        throw std::runtime_error("Batch size B must equal 1 in inference mode.");
    }
    
    model->forward(out, in);

    float* probabilities = out->act + (t-1)*V;
    uint16_t next_tok = sample_token(probabilities, V, true);

    return next_tok;
}

void yap(GPT2*& model,
         Tokenizer*& tk,
         Node*& out,
         Node*& in,
         std::string start){

    // clear kv cache
    model->clear_kv_cache();

    // maybe check / assert B = 1


    size_t T = in->shape[1];
    uint16_t eot = 50256;
    size_t t; // for next_token call

    std::string decoded_tokens = start;

    if (start.empty() ){

        for (size_t i = 0; i < T; i++){
            in->act[i] = eot;
        }

        t = 1;
    }

    else {

        std::vector<uint16_t> encoded = tk->encode(start);
        size_t num_tokens = encoded.size();

        for (size_t i = 0; i < num_tokens; i++){
            in->act[i] = encoded[i];
        }

        for (size_t i = num_tokens; i < T; i++){
            in->act[i] = eot;
        }

        t = num_tokens;

    }

    // autogenerate tokens
    std::cout << start;
    for (size_t i = t; i < T; i++){
        uint16_t next_tok = next_token(model, out, in, i);
        
        in->act[i] = next_tok;
        std::string next_tok_dec = tk->decode({next_tok});
        std::cout << next_tok_dec << std::flush;

        decoded_tokens += next_tok_dec;
    }

    // clear kv cache
    model->clear_kv_cache();
}