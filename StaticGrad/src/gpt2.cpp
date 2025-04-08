#include "gpt2.hpp"
#include <fstream>
#include <cstring>
#include <cmath>

/**
 * @brief Calculate the number of external activations required for a given batch size, sequence length, and model configuration.
 *
 * @param B Batch size.
 * @param T Sequence length.
 * @param C Number of attention heads.
 * @param L Number of transformer blocks.
 * @param V Size of the vocabulary.
 * @return The number of activations.
 * 
 * @note This figure does not include the internal activations for the mean (B*T) and rstd (B*T) used by layernorms and the internal pre_softmax and post_softmax buffers used
 * in an attention block.
 */
size_t gpt2_num_acts(const size_t B,
                     const size_t T,
                     const size_t C,
                     const size_t L,
                     const size_t V)
{
    size_t num_acts = B*T;

    // encoder (B, T) -> (B, T, C)
    num_acts += B*T*C;

    // transformer blocks (B, T, C) -> (B, T, C)
    num_acts += L * (16 * B * T * C);

    // final layernorm (B, T, C) -> (B, T, C)
    num_acts += B*T*C;

    // unembedding (B, T, C) -> (B, T, V)
    num_acts += B*T*V;

    return num_acts;
};

/**
 * @brief Calculate the number of parameters for a given model configuration.
 *
 * @param C Number of attention heads.
 * @param L Number of transformer blocks.
 * @param vocab_size Size of the vocabulary.
 * @param max_seqlen Maximum sequence length.
 * @return The number of parameters in the model
 */
size_t gpt2_memrequirement(const size_t C,
                           const size_t L,
                           const size_t vocab_size,
                           const size_t max_seqlen)
{

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

 /**
  * Constructor for the GPT2 class.
  *
  * @param C_ Input size for the embeddings
  * @param L_ Number of layers in the transformer
  * @param V_ Vocabulary size of the tokenizer
  * @param maxT_ Maximum sequence length
  * @param NH_ Number of heads in the self-attention layers 
  * 
  * @throws `std::runtime_error` if parameter allocation is done incorrectly
  */
GPT2::GPT2(const size_t C_,
           const size_t L_,
           const size_t V_,
           const size_t maxT_,
           const size_t NH_):
    Operation(nullptr, nullptr),
    C(C_),
    L(L_),
    V(V_),
    maxT(maxT_),
    NH(NH_)
    {
        num_params = gpt2_memrequirement(C, L, V, maxT);

        params = new float[num_params];
        grad = new float[num_params];

        float* p = params;
        float* g = grad;

        encoder = new Encoder(p, g, C, V);
        p += V*C + maxT*C;
        g += V*C + maxT*C;


        for (size_t l = 0 ; l < L; l++){
            tblocks.push_back(new TransformerBlock(p, g, C, NH));
            p += (12*C*C + 13*C);
            g += (12*C*C + 13*C);
        }

        final_layernorm = new LayerNorm(p, g);
        p += C + C;
        g += C + C;

        unembedding = new Matmul(params, grad, 1.0f, false);

        if (p - params != static_cast<int>(num_params) || g - grad != static_cast<int>(num_params)){
            throw std::runtime_error("parameter allocation incorrect");
        }

}

/**
 * @brief Destroy the GPT2 object. Deletes dynamically allocated memory for the model parameters, gradients, and submodules.
 */
GPT2::~GPT2(){
    delete unembedding;
    delete final_layernorm;

    for (size_t l = 0; l < L; l++){
        delete tblocks[l];
    }

    delete encoder;
    
    delete[] grad;
    delete[] params;
}

/**
 * 
 * @brief Clear the key-value caches and their gradient buffers of all transformer blocks
 * 
 */
void GPT2::clear_kv_cache(){
    for (auto& tblock : tblocks){
        delete[] tblock->att->buffer;
        delete[] tblock->att->dbuffer;
        tblock->att->buffer = nullptr;
        tblock->att->dbuffer= nullptr;
    }
}



/**
 * @brief Performs the forward pass of the GPT-2 model.
 *
 * This function transforms the input sequence tensor of shape (B, T) to an output tensor 
 * of shape (B, T, V), where B is the batch size, T is the sequence length, and V is the 
 * vocabulary size. The forward pass involves the following major components:
 * 
 * - Encoder: Token and positional embedding of sequence into latent space.
 * 
 * - Transformer Blocks: A sequence of self-attention and feed-forward neural networks that process the encoded input.
 * 
 * - Layer Normalization: Applies layer normalization to the output of the last transformer block.
 * 
 * - Unembedding: Maps from latent space back to token space.
 * 
 * @param out   The output Node, containing the logits over vocabulary for each input sequence.
 *              Shape: (B, T, V), where B is the batch size, T is the sequence length, and V is the vocabulary size.
 * @param in    The input Node, containing the token indices of the input sequences. Shape: (B, T).
 *
 * @throws `std::runtime_error` if the memory allocation for activations is not managed properly.
 */
void GPT2::forward(Node* out, Node* in){
    // in is shape (B, T)
    // out is shape (B, T, V)

    size_t B = in->shape[0];
    size_t T = in->shape[1];
    
    // forward through the encoder 
    Node* in_internal = new Node();
    in_internal->act = in->act;
    in_internal->act_grads = in->act_grads;
    in_internal->shape = in->shape;
    in_internal->size = in->size;
    in_internal->current_T = in->current_T;

    Node* out_internal = new Node();
    out_internal->act = in_internal->act + B * T;
    out_internal->act_grads = in_internal->act_grads + B * T; 
    out_internal->shape = {B, T, C}; // output shape of encoder
    out_internal->size = B * T * C;
    out_internal->current_T = in->current_T;


    encoder->forward(out_internal, in_internal);

    // forward through the transformer blocks
    for (size_t i = 0; i < L; i++){
        TransformerBlock* tblock = tblocks[i];

        // set up in_internal and out_internal
        in_internal->act = out_internal->act;
        in_internal->act_grads = out_internal->act_grads;
        in_internal->shape = out_internal->shape;
        in_internal->size = out_internal->size;

        out_internal->act += 16 * B*T*C; // 16*B*T*C is the number of activations used in a tblock
        out_internal->act_grads += 16 * B*T*C;
        out_internal->shape = {B, T, C};
        out_internal->size = B*T*C;

        // forward through i-th TransformerBlock
        tblock->forward(out_internal, in_internal);
    }

    // forward through layernorm
    shift(out_internal, in_internal, {B, T, C});
    final_layernorm->forward(out_internal, in_internal);

    // forward through unembedding (matmul)
    shift(out_internal, in_internal, {B, T, V});
    unembedding->forward2(out_internal, in_internal); // (B, T, C) - > (B, T, V)

    // verify that results are in out Node
    if ((out_internal->act != out->act) || (out_internal->act_grads != out->act_grads) || (out_internal->size != out->size)){
        throw std::runtime_error("out node and out_internal node are not equal");
    }

    delete in_internal;
    delete out_internal;

}

/**
 * @brief Performs the backward pass of the GPT-2 model.
 *
 * This function backpropagates gradients from the loss through the layers of the model.
 * 
 * @param out   The output Node, containing the logits over vocabulary for each input sequence.
 *              Shape: (B, T, V), where B is the batch size, T is the sequence length, and V is the vocabulary size.
 * @param in    The input Node, containing the token indices of the input sequences. Shape: (B, T).
 *
 * @throws `std::runtime_error` if the memory allocation for activations is not managed properly.
 */
void GPT2::backward(Node* out, Node* in){
    // in is shape (B, T)
    // out is shape (B, T, V)

    size_t B = in->shape[0];
    size_t T = in->shape[1];

    Node* out_internal = new Node();
    out_internal->act = out->act;
    out_internal->act_grads = out->act_grads;
    out_internal->shape = out->shape; 
    out_internal->size = out->size;

    Node* in_internal = new Node();
    in_internal->shape = {B, T, C};
    in_internal->size = B*T*C;
    in_internal->act = out->act - in_internal->size;
    in_internal->act_grads = out->act_grads - in_internal->size;

    unembedding->backward2(out_internal, in_internal);

    // backward through layernorm
    shift_back(out_internal, in_internal, {B, T, C});
    final_layernorm->backward(out_internal, in_internal);

    // backward through the transformer blocks 
    for (int i = L-1; i >=0 ; i--){  // the loop variable i must be int not size_t, otherwise it will crash the code
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
    }

    // backwards through encoder
    shift_back(out_internal, in_internal, {B, T});
    encoder->backward(out_internal, in_internal);

    // verify that results are in out Node
    if ((in_internal->act != in->act) || (in_internal->act_grads != in->act_grads) || (in_internal->size != in->size)){
        throw std::runtime_error("in node and in_internal node are not equal");
    }

    delete in_internal;
    delete out_internal;
}


/**
 * Loads weights from a file into the models parameters.
 *
 * @param fname: The name of the file containing the weights.
 *
 * @throws `std::runtime_error` if the file cannot be opened, the number of bytes in the file does not match the expected number, or an error occurs while reading the file.
 * 
 */
void GPT2::load_weights(const std::string& fname)
{
    // Open the file in binary mode
    std::ifstream inputFile(fname, std::ios::binary);

    if (!inputFile.is_open()){
        throw std::runtime_error("could not open file");
    }

    // Get the size of the file
    inputFile.seekg(0, std::ios::end);
    size_t fileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    if ((this->num_params) * sizeof(float) != fileSize){
        throw std::runtime_error("number of bytes in file does not match expected");
    }
    

    // Read the contents of the file into the buffer
    if (!inputFile.read(reinterpret_cast<char*>(this->params), fileSize)) {
        throw std::runtime_error("error loading contents into buffer");
    }
}