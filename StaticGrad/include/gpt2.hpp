#ifndef GPT2_HPP
#define GPT2_HPP

#include "./classes.hpp"

// get number of activations needed for forward pass
size_t gpt2_num_acts(size_t B, size_t T, size_t C, size_t L, size_t V);

// get number of parameters to malloc for gpt2 parameters
size_t gpt2_memrequirement(size_t C, size_t L, size_t vocab_size, size_t max_seqlen);

class GPT2 {
    public:

        GPT2(size_t C_, size_t L_, size_t V_, size_t maxT_, size_t NH_); // parameterized constructor

        GPT2(): // default constructor.
            GPT2(768, 12, 50257, 1024, 12) {} 

        ~GPT2();

        void zero_grad();
        void set_temperature(float temp);
        void update(int t); 

        void forward(Node* out, Node* in);
        void backward(Node* out, Node* in);

        const size_t C;    // embedding dimension (default 768)
        const size_t L;    // number of transformer blocks (default 12)
        const size_t V;    // vocab size (default 50257)
        const size_t maxT; // max sequence length (default 1024)
        const size_t NH; // number of attention heads

        size_t num_params; // number of parameters

        Encoder* encoder; // encoder
        std::vector<TransformerBlock*> tblocks; // vector of TransformerBlocks
        LayerNorm* final_layernorm; // final layer norm
        Matmul* unembedding; // unembedding 
        Softmax* softmax; // softmax

        float* params;
        float* grad;


        float* m; // first moment for adam
        float* v; // second moment for adam
        float beta1;
        float beta2;
        float alpha; // learn rate

};

#endif // GPT2_HPP