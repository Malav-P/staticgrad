#include "tokenizer.hpp"
#include "gpt2.hpp"
#include "inference.hpp"
#include "utils.hpp"

/**
 * Computes the most probable token at specified position
 *
 * Args:
 *   @param model: Pointer to instance of GPT2
 *   @param tk: Pointer to instance of tokenizer
 *   @param out: The output node where the computed loss will be stored, a 2D tensor of shape (B, T).
 *   @param in: The input node containing the probabilities, a 3D tensor of shape (B, T, V).
 *   @param t: Desired position of next token in sequence
 *
 * Returns:
 *   u_int16_t. The most probable token for that position
 */
u_int16_t next_token(GPT2*& model,
                     Tokenizer*& tk,
                     Node*& out,
                     Node*& in,
                     size_t t){

    // ENSURE THAT t is NEVER ZERO, OTHERWISE UNDERFLOW ERROR WILL OCCUR

    size_t B = out->shape[0];
    size_t T = out->shape[1];
    size_t V = out->shape[2];

    if (B != 1){
        throw std::runtime_error("Batch size B must equal 1 in inference mode.");
    }
    
    model->forward(out, in);

    float* probabilities = out->act + (t-1)*V;
    u_int16_t next_token = sample_token(probabilities, V, false);

    return next_token;

}