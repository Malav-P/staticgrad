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

    // for debugging purposes, can remove
    // std::cout << "encoder starting...\n";

    
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
    unembedding->forward(out_internal, in_internal); // (B, T, C) - > (B, T, V)

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
    softmax->forward(out_internal, in_internal); // (B, T, V) -> (B, T, V)

    // for debugging, can remove
    // std::cout << "softmax complete...\n";


    // verify that results are in out Node
    if ((out_internal->act != out->act) || (out_internal->act_grads != out->act_grads) || (out_internal->size != out->size)){
        throw std::runtime_error("out node and out_internal node are not equal");
    }

    delete in_internal;
    delete out_internal;

}

void GPT2::backward(Node* out, Node* in){
    // in is shape (B, T)
    // out is shape (B, T, V)

    size_t B = in->shape[0];
    size_t T = in->shape[1];

    Node* out_internal = new Node();
    out_internal->act = out->act;
    out_internal->act_grads = out->act_grads; // should remove B*T since input does not require grads. Here for readability
    out_internal->shape = out->shape; // output shape of encoder
    out_internal->size = out->size;

    Node* in_internal = new Node();
    in_internal->act = out_internal->act - B*T*V;
    in_internal->act_grads = out_internal->act_grads - B*T*V;
    in_internal->shape = {B, T, V};
    in_internal->size = B*T*V;

    // for debugging, can remove
    std::cout << "beginning softmax...\n";

    // backward through softmax takes forever, need to couple with cross entropy loss.
    // TODO make sure the data here is written already by another function outside of this method. name
    // the other function crossentropy_softmax_backward()
    // softmax->backward(out_internal, in_internal);

    // for debugging, can remove
    std::cout << "softmax complete...\n";

    // set up out_internal and in_internal
    out_internal->act = in_internal->act;
    out_internal->act_grads = in_internal->act_grads;
    out_internal->shape = in_internal->shape;
    out_internal->size = in_internal->size;

    in_internal->size = B*T*C;
    in_internal->shape = {B, T, C};
    in_internal->act -= in_internal->size;
    in_internal->act_grads -= in_internal->size;

    //backward through unembedding
    unembedding->backward(out_internal, in_internal);

    // for debugging, can remove
    std::cout << "unembedding complete...\n";

    // set up out_internal and in_internal
    out_internal->act = in_internal->act;
    out_internal->act_grads = in_internal->act_grads;
    out_internal->shape = in_internal->shape;
    out_internal->size = in_internal->size;

    in_internal->size = B*T*C;
    in_internal->shape = {B, T, C};
    in_internal->act -= in_internal->size;
    in_internal->act_grads -= in_internal->size;

    // backward through layernorm
    final_layernorm->backward(out_internal, in_internal);

    // backward through the transformer blocks
    for (int i = L-1; i >=0 ; i--){
        TransformerBlock* tblock = tblocks[i];

        // set up in_internal and out_internal
        out_internal->act = in_internal->act;
        out_internal->act_grads = in_internal->act_grads;
        out_internal->shape = in_internal->shape;
        out_internal->size = in_internal->size;

        in_internal->act -= 16 * B*T*C; // 16*B*T*C is the number of activations used in a tblock
        in_internal->act_grads -= 16*B*T*C;
        in_internal->shape = {B, T, C};
        in_internal->size = B*T*C;

        // backward through i-th TransformerBlock
        tblock->backward(out_internal, in_internal);

        // for debugging purposes, can remove
        std::cout<< "tblock " << i + 1 << " complete\n";
    }

    // set up out_internal and in_internal
    out_internal->act = in_internal->act;
    out_internal->act_grads = in_internal->act_grads;
    out_internal->shape = in_internal->shape;
    out_internal->size = in_internal->size;

    in_internal->size = B*T;
    in_internal->shape = {B, T};
    in_internal->act -= in_internal->size;
    in_internal->act_grads -= in_internal->size;

    // backwards through encoder
    encoder->backward(out_internal, in_internal);

    // verify that results are in out Node
    if ((in_internal->act != in->act) || (in_internal->act_grads != in->act_grads) || (in_internal->size != in->size)){
        throw std::runtime_error("in node and in_internal node are not equal");
    }

    delete in_internal;
    delete out_internal;

}