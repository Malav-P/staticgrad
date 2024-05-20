#ifndef GPT2_HPP
#define GPT2_HPP

#include "./classes.hpp"

// get number of activations needed for forward pass
size_t gpt2_num_acts(size_t B, size_t T, size_t C, size_t L, size_t V);

// get number of bytes to malloc for gpt2 parameters
size_t gpt2_memrequirement(size_t C, size_t L, size_t vocab_size, size_t max_seqlen);

class GPT2 {
    public:

        GPT2(size_t C_, size_t L_, size_t V_, size_t maxT_, size_t NH_):
            C(C_),
            L(L_),
            V(V_),
            maxT(maxT_),
            NH(NH_){
                num_params = gpt2_memrequirement(C, L, V, maxT);

                params = new float[num_params];
                grad = new float[num_params];

                float* p = params;
                float* g = grad;

                encoder = new Encoder(p, g, C, V);
                p += V*C + maxT*C;
                g += V*C + maxT*C;


                for (int l = 0 ; l < L; l++){
                    tblocks.push_back(new TransformerBlock(p, g, C, NH, maxT));
                    p += (12*C*C + 13*C);
                    g += (12*C*C + 13*C);
                }

                final_layernorm = new LayerNorm(p, g);
                p += C + C;
                g += C + C;

                unembedding = new Matmul(params, grad);

                softmax = new Softmax();

                if (p - params != num_params || g - grad != num_params){
                    throw std::runtime_error("parameter allocation incorrect");
                }

            }

        ~GPT2(){
            delete softmax;
            delete unembedding;
            delete final_layernorm;

            for (int l = 0; l < L; l++){
                delete tblocks[l];
            }

            delete encoder;

            delete[] grad;
            delete[] params;
        }

        void zero_grad(){
            std::memset(grad, 0, num_params * sizeof(float));
        }


        void forward(Node* out, Node* in);

        size_t C;    // embedding dimension (default 768)
        size_t L;    // number of transformer blocks (default 12)
        size_t V;    // vocab size (default 50257)
        size_t maxT; // max sequence length (default 1024)
        size_t NH; // number of attention heads

        size_t num_params;

        Encoder* encoder; // encoder
        std::vector<TransformerBlock*> tblocks; // vector of TransformerBlocks
        LayerNorm* final_layernorm; // final layer norm
        Matmul* unembedding; // unembedding 
        Softmax* softmax; // softmax

        float* params;
        float* grad;

};

#endif // GPT2_HPP