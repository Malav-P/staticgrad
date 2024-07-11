#include "datastream.hpp"
#include "tokenizer.hpp"
#include "gpt2.hpp"
#include "interface.hpp"
#include "utils.hpp"


/**
* Allocates and initializes the GPT-2 model, data stream, tokenizer, and nodes for a specified batch size and sequence length.
*
*  @param model  A pointer (passed by reference) to a GPT-2 model that will be allocated and initialized.
*  @param ds A pointer (passed by reference) to a DataStream that will be allocated and initialized.
*  @param tk A pointer (passed by reference) to a Tokenizer that will be allocated and initialized.
*  @param out A pointer (passed by reference) to an output node that will be allocated and initialized.
*  @param in  A pointer (passed by reference) to an input node that will be allocated and initialized.
*  @param B The batch size.
*  @param T The sequence length.
*  @param pretrained If true, load pre-trained weights for the model.
*
*  
*  @throws `std::invalid_argument` If any of the passed pointers are not nullptr.
*
*  
*   @note Postconditions:
*   @note - The model, data stream, tokenizer, and nodes are allocated and initialized.
*   @note - The model is loaded with pre-trained weights if pretrained is true.
*   @note - The data stream is initialized with a buffer of size B*T + 1.
*   @note - The tokenizer is initialized with the vocabulary from GPT-2.
*   @note - The input and output nodes are allocated and have appropriate shapes and sizes.
*/
void setup(GPT2*& model,
           DataStream*& ds,
           Tokenizer*& tk,
           Node*& out, 
           Node*& in,
           size_t B, // batch size
           size_t T,
           bool pretrained) // sequence length
{
    
    if (model != nullptr){
        throw std::invalid_argument("model must be empty (nullptr)");
    }

    if (ds != nullptr){
        throw std::invalid_argument("datastream must be empty (nullptr)");
    }

    if (tk != nullptr){
        throw std::invalid_argument("tokenizer must be empty (nullptr)");
    }

    if (out != nullptr){
        throw std::invalid_argument("out node must be empty (nullptr)");
    }

    if (in != nullptr){
        throw std::invalid_argument("in node must be empty (nullptr)");
    }

    size_t C = 768; // embedding dimension
    size_t L = 12; // number of transformer blocks
    size_t V = 50257; // vocab size
    size_t maxT = 1024; // max sequence length
    size_t NH = 12; // number of attention heads

    model = new GPT2(C, L, V, maxT, NH);

    if (pretrained){
        int expected_bytes = sizeof(float) * 124439808;
        std::string weights_file = "/Users/malavpatel/Coding_Projects/StaticGrad/models/gpt2.bin";
        load_weights(model->params, weights_file, expected_bytes);

        // The gpt2.bin file represents weight token embedding matrix as shape (V, C). In this code, we assume the weight token embedding matrix
        // also has shape (V, C)
    }

    std::string fp_ds = "/Users/malavpatel/Coding_Projects/StaticGrad/tokenizer/tokens/tinystories.bin";
    ds = new DataStream();
    ds->open(fp_ds);
    ds->init_buffer(B*T + 1); // +1 to include necessary target tokens for training

    std::string fp_tk = "/Users/malavpatel/Coding_Projects/StaticGrad/tokenizer/gpt2_vocab.bin";
    tk = new Tokenizer(fp_tk);

    size_t num_acts = gpt2_num_acts(B, T, C, L, V);

    float* acts = new float[num_acts];
    float* grad = new float[num_acts];

    in = new Node();
    in->act = acts;
    in->act_grads = grad;
    in->shape = {B, T};
    in->size = B*T;

    out = new Node();
    out->act = acts + num_acts - B*T*V;
    out->act_grads = grad + num_acts - B*T*V;
    out->shape = {B, T, V};
    out->size = B*T*V;

    std::cout << "\nsetup complete, gpt2 model instantiated with " << model->num_params << " parameters and " << num_acts + L*2*B*T + L*NH*B*T*(T+1) + 2*B*T << " activations" << std::endl;    

}


/**
* Deallocates the memory allocated by the setup function.
*
* @param model: A pointer (passed by reference) to the GPT-2 model to be deallocated.
* @param ds: A pointer (passed by reference) to the DataStream to be deallocated. 
* @param tk: A pointer (passed by reference) to the Tokenizer to be deallocated.
* @param out: A pointer (passed by reference) to the output node to be deallocated.
* @param in: A pointer (passed by reference) to the input node to be deallocated.
*
* @note Postconditions:
*   @note - All memory allocated by the setup function is released.
*   @note - All passed pointers are set to nullptr.
*/
void tear_down(GPT2*& model,
               DataStream*& ds,
               Tokenizer*& tk,
               Node*& out, 
               Node*& in)
{
    delete model;
    model = nullptr;

    delete ds;
    ds = nullptr;

    delete tk;
    tk = nullptr;

    delete[] in->act;
    delete[] in->act_grads;
    delete in;
    in = nullptr;

    delete out;
    out = nullptr;

    std::cout << "\nteardown complete, memory deallocated" << std::endl;
}

/**
* Trains the GPT-2 model for a specified number of batches.
*
* @param max_batches: The number of batches to train for.
*
* @throws `std::runtime_error` If the setup function has not been called before train.
* 
* @note The GPT-2 model, data stream, tokenizer, in, and out nodes are properly initialized through the `setup` function
*/
void train(int max_batches){
    // setup model
    GPT2* model = nullptr;
    DataStream* ds = nullptr;
    Tokenizer* tk = nullptr;
    Node* out = nullptr;
    Node* in = nullptr;

    size_t B = 4;
    size_t T = 64;

    bool pretrained = true;

    setup(model, ds, tk, out, in, B, T, pretrained);


    // create node for storing losses.
    Node* loss = new Node();
    loss->shape = {B, T};
    loss->size = B*T;
    loss->act = new float[B*T];
    loss->act_grads = new float[B*T];

    // create a node for softmax in
    Node* softmax_in = new Node();
    softmax_in->shape = {B, T, model->V};
    softmax_in->size = B*T*model->V;
    softmax_in->act = out->act - B*T*model->V;
    softmax_in->act_grads = out->act_grads - B*T*model->V;

    // number of external activations used (exludes internal activates like m, rstd used in layernorm and buffers in attention)
    size_t num_acts = gpt2_num_acts(B, T, model->C, model->L, model->V);

    // begin training
    for (int t = 0; t < max_batches; t++){
        ds->load_buffer(); // loads B*T + 1 tokens

        // std::string first_str = tk->decode(ds->buffer, T);
        // std::string first_tar = tk->decode((ds->buffer) + 1, T);
        // std::cout << "FIRST SEQUENCE: " << first_str << std::endl;
        // std::cout << "FIRST TARGETS: " << first_tar << std::endl;

        ds->buffer_to_Node(in, B*T); // transfer first B*T tokens from buffer to in node

        model->forward(out, in); // do forward pass

        crossentropy_forward(loss, out, (ds->buffer) + 1); // compute loss

        model->zero_grad(); // zero gradients of parameters
        std::memset(in->act_grads, 0, sizeof(float)*num_acts); // set gradient of activations to zero.

        crossentropy_softmax_backward(out, softmax_in, (ds->buffer) + 1, model->softmax->temperature); // do backward through crossentropy loss and softmax
        model->backward(out, in);
        model->update(t+1);

        float loss_val = 0.0f;
        for (int i = 0; i < B*T; i++){
            loss_val += loss->act[i];
        }
        loss_val = loss_val / (B*T);

        std::cout << "iteration " << t << ": loss = " << loss_val << std::endl;
    }


    delete[] loss->act;
    delete[] loss->act_grads;
    delete loss;

    delete softmax_in;

    tear_down(model, ds, tk, out, in);
}


