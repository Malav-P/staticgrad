#include "./classes.hpp"

class GPT2 {
    public:

        void forward(Node* out, Node* in);

        int C;    // embedding dimension (default 768)
        int L;    // number of transformer blocks (default 12)
        int V;    // vocab size (default 50267)
        int maxT; // max sequence length (default 1024)

};

// get number of bytes to malloc for gpt2 parameters
int gpt2_memrequirement(int C, int L, int vocab_size, int max_seqlen){

    int num_params = 0;

    // ------TRANSFORMER BLOCK----------

    // first layer norm
    num_params += C + C; 

    // matmul from attention (Q, K, V) matrices and bias;
    num_params += C * 3*C + 3*C;

    // matmul from post attention
    num_params += C*C + C;

    // second layer norm
    num_params += C + C;

    // fully connected layer (projection into higher dimension)
    num_params += C * 4*C + 4*C;

    // second fully connected layer (projection back into embedding dimension)
    num_params += 4*C * C + C;

    // --------------------------------

    // multiply by number of transformer blocks
    num_params *= L;

    // token embedding matrix
    num_params += C * vocab_size;

    // positional embedding matrix
    num_params += C * max_seqlen;

    // final layer norm
    num_params += C + C;


    // multiply by size of float

    int memory_req = sizeof(float) * num_params;

    return memory_req;

}