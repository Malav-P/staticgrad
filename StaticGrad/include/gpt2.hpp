#ifndef GPT2_HPP
#define GPT2_HPP

#include <string>
#include "classes.hpp"

// get number of activations needed for forward pass
size_t gpt2_num_acts(const size_t B, const size_t T, const size_t C, const size_t L, const size_t V);

// get number of parameters to malloc for gpt2 parameters
size_t gpt2_memrequirement(const size_t C, const size_t L, const size_t vocab_size, const size_t max_seqlen);

class GPT2 : public Operation {
    public:

        GPT2(size_t C_, size_t L_, size_t V_, size_t maxT_, size_t NH_); // parameterized constructor

        GPT2(): // default constructor.
            GPT2(768, 12, 50257, 1024, 12) {} 

        ~GPT2();

        void load_weights(const std::string& fname);

        void forward(Node* out, Node* in);
        void backward(Node* out, Node* in);

        void clear_kv_cache();

        void train_mode();

        const size_t C;    // embedding dimension (default 768)
        const size_t L;    // number of transformer blocks (default 12)
        const size_t V;    // vocab size (default 50257)
        const size_t maxT; // max sequence length (default 1024)
        const size_t NH;   // number of attention heads

        size_t num_params; // number of parameters
        size_t num_bytes; // number of bytes parameters use

        Embedding* encoder; // encoder
        std::vector<TransformerBlock*> tblocks; // vector of TransformerBlocks
        LayerNorm* final_layernorm; // final layer norm
        Matmul* unembedding; // unembedding 

        Node* input;
        Node* output;

};

#endif // GPT2_HPP