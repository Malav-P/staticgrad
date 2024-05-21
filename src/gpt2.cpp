#include "./gpt2.hpp"


size_t gpt2_num_acts(size_t B, size_t T, size_t C, size_t L, size_t V){
    size_t num_acts = B*T;

    // encoder (B, T) -> (B, T, C)
    num_acts += B*T*C;

    // transformer blocks (B, T, C) -> (B, T, C)
    num_acts += L * (16 * B * T * C);

    // final layernorm (B, T, C) -> (B, T, C)
    num_acts += B*T*C;

    // unembedding (B, T, C) -> (B, T, V)
    num_acts += B*T*V;

    // softmax (B, T, V) -> (B, T, V)
    num_acts += B*T*V;

    return num_acts;
};

size_t gpt2_memrequirement(size_t C, size_t L, size_t vocab_size, size_t max_seqlen){

    size_t num_params = 0;

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


    return num_params;

};

void GPT2::forward(Node* out, Node* in){
    // in is shape (B, T)
    // out is shape (B, T, V)

    size_t B = in->shape[0];
    size_t T = in->shape[1];

    Node* in_internal = new Node();
    in_internal->act = in->act;
    in_internal->act_grads = in->act_grads;
    in_internal->shape = in->shape;
    in_internal->size = in->size;

    Node* out_internal = new Node();
    out_internal->act = in_internal->act + B * T;
    out_internal->act_grads = in_internal->act_grads + B * T; // should remove B*T since input does not require grads. Here for readability
    out_internal->shape = {B, T, C}; // output shape of encoder
    out_internal->size = B * T * C;
    
    // forward through the encoder 
    encoder->forward(out_internal, in_internal);

    // for debugging purposes, can remove
    // std::cout << "encoder complete...\n";

    // forward through the transformer blocks
    for (int i = 0; i < L; i++){
        TransformerBlock* tblock = tblocks[i];

        // set up in_internal and out_internal
        in_internal->act = out_internal->act;
        in_internal->act_grads = out_internal->act_grads;
        in_internal->shape = out_internal->shape;
        in_internal->size = out_internal->size;

        out_internal->act += 16 * B*T*C; // 16*B*T*C is the number of activations used in a tblock
        out_internal->act_grads += 16*B*T*C;
        out_internal->shape = {B, T, C};
        out_internal->size = B*T*C;

        // forward through i-th TransformerBlock
        tblock->forward(out_internal, in_internal);

        // for debugging purposes, can remove
        // std::cout<< "tblock " << i + 1 << " complete\n";
    }

    // set up in_internal and out_internal
    in_internal->act = out_internal->act;
    in_internal->act_grads = out_internal->act_grads;
    in_internal->shape = out_internal->shape;
    in_internal->size = out_internal->size;

    out_internal->act += out_internal->size;
    out_internal->act_grads += out_internal->size;
    out_internal->shape = {B, T, C};
    out_internal->size = B*T*C;

    // forward through layernorm
    final_layernorm->forward(out_internal, in_internal);

    // for debugging, can remove
    // std::cout << "final layernorm complete...\n";

    // set up in_internal and out_internal
    in_internal->act = out_internal->act;
    in_internal->act_grads = out_internal->act_grads;
    in_internal->shape = out_internal->shape;
    in_internal->size = out_internal->size;

    out_internal->act += out_internal->size;
    out_internal->act_grads += out_internal->size;
    out_internal->shape = {B, T, V};
    out_internal->size = B*T*V;

    // forward through unembedding (matmul)
    unembedding->forward(out_internal, in_internal);

    // for debugging, can remove
    // std::cout << "unembedding complete...\n";


    // set up in_internal and out_internal
    in_internal->act = out_internal->act;
    in_internal->act_grads = out_internal->act_grads;
    in_internal->shape = out_internal->shape;
    in_internal->size = out_internal->size;

    out_internal->act += out_internal->size;
    out_internal->act_grads += out_internal->size;
    out_internal->shape = {B, T, V};
    out_internal->size = B*T*V;

    // forward through softmax
    softmax->forward(out_internal, in_internal);

    // for debugging, can remove
    // std::cout << "softmax complete...\n";


    // verify that results are in out Node
    if ((out_internal->act != out->act) || (out_internal->act_grads != out->act_grads) || (out_internal->size != out->size)){
        throw std::runtime_error("out node and out_internal node are not equal");
    }

    delete in_internal;
    delete out_internal;

}