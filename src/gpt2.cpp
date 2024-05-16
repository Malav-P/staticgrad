#include "./gpt2.hpp"

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

    // forward through the transformer blocks
    for (int i = 0; i < L; i++){
        TransformerBlock* tblock = tblocks[i];

        // set up in_internal and out_internal
        in_internal->act = out_internal->act;
        in_internal->act_grads = out_internal->act_grads;
        in_internal->shape = out->shape;
        in_internal->size = out->size;

        out_internal->act += 16 * B*T*C; // 16*B*T*C is the number of activations used in a tblock
        out_internal->act_grads += 16*B*T*C;
        out_internal->shape = {B, T, C};
        out_internal->size = B*T*C;

        // forward through i-th TransformerBlock
        tblock->forward(out_internal, in_internal);
    }

    // set up in_internal and out_internal
    in_internal->act = out_internal->act;
    in_internal->act_grads = out_internal->act_grads;
    in_internal->shape = out->shape;
    in_internal->size = out->size;

    out_internal->act += out_internal->size;
    out_internal->act_grads += out_internal->size;
    out_internal->shape = {B, T, C};
    out_internal->size = B*T*C;

    // forward through layernorm
    final_layernorm->forward(out_internal, in_internal);

    // set up in_internal and out_internal
    in_internal->act = out_internal->act;
    in_internal->act_grads = out_internal->act_grads;
    in_internal->shape = out->shape;
    in_internal->size = out->size;

    out_internal->act += out_internal->size;
    out_internal->act_grads += out_internal->size;
    out_internal->shape = {B, T, V};
    out_internal->size = B*T*V;

    // forward through unembedding (matmul)
    unembedding->forward(out_internal, in_internal);

    // set up in_internal and out_internal
    in_internal->act = out_internal->act;
    in_internal->act_grads = out_internal->act_grads;
    in_internal->shape = out->shape;
    in_internal->size = out->size;

    out_internal->act += out_internal->size;
    out_internal->act_grads += out_internal->size;
    out_internal->shape = {B, T, V};
    out_internal->size = B*T*V;

    // forward through softmax
    softmax->forward(out_internal, in_internal);

    // verify that results are in out Node
    if ((out_internal->act != out->act) || (out_internal->act_grads != out->act_grads) || (out_internal->size != out->size)){
        throw std::runtime_error("out node and out_internal node are not equal");
    }

}