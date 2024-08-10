#include "datastream.hpp"
#include "tokenizer.hpp"
#include "gpt2.hpp"
#include "interface.hpp"
#include "utils.hpp"

const std::string PREFIX = REPO_PREFIX;

/**
 * Computes the most probable token at specified position
 *
 * Args:
 *   @param model: Pointer to instance of GPT2
 *   @param out: The output node where the computed loss will be stored, a 2D tensor of shape (B, T).
 *   @param in: The input node containing the probabilities, a 3D tensor of shape (B, T, V).
 *   @param t: Desired position of next token in sequence
 *
 * Returns:
 *   uint16_t. The most probable token for that position
 */
uint16_t next_token(GPT2*& model,
                     Node*& out,
                     Node*& in,
                     const size_t t){

    if (t == 0){
        throw std::runtime_error("t cannot be zero");
    }

    size_t B = out->shape[0];
    size_t V = out->shape[2];

    in->current_T = t;

    if (B != 1){
        throw std::runtime_error("Batch size B must equal 1 in inference mode.");
    }
    
    model->forward(out, in);

    float* probabilities = out->act + (t-1)*V;
    uint16_t next_tok = sample_token(probabilities, V, true);

    return next_tok;
}


void yap(GPT2*& model,
         Tokenizer*& tk,
         Node*& out,
         Node*& in,
         const std::string start){

    // clear kv cache
    model->clear_kv_cache();

    // maybe check / assert B = 1


    size_t T = in->shape[1];
    uint16_t eot = 50256;
    size_t t; // for next_token call

    std::string decoded_tokens = start;

    if (start.empty() ){

        for (size_t i = 0; i < T; i++){
            in->act[i] = eot;
        }

        t = 1;
    }

    else {

        std::vector<uint16_t> encoded = tk->encode(start);
        size_t num_tokens = encoded.size();

        for (size_t i = 0; i < num_tokens; i++){
            in->act[i] = encoded[i];
        }

        for (size_t i = num_tokens; i < T; i++){
            in->act[i] = eot;
        }

        t = num_tokens;

    }

    // autogenerate tokens
    std::cout << start;
    for (size_t i = t; i < T; i++){
        uint16_t next_tok = next_token(model, out, in, i);
        
        in->act[i] = next_tok;
        std::string next_tok_dec = tk->decode({next_tok});
        std::cout << next_tok_dec << std::flush;

        decoded_tokens += next_tok_dec;
    }

    // clear kv cache
    model->clear_kv_cache();
}


void setup_model(GPT2*& model, const bool pretrained){
    if (model != nullptr){
        throw std::runtime_error("passed model pointer must be null");
    }

    const size_t C = 768; // embedding dimension
    const size_t L = 12; // number of transformer blocks
    const size_t V = 50257; // vocab size
    const size_t maxT = 1024; // max sequence length
    const size_t NH = 12; // number of attention heads

    model = new GPT2(C, L, V, maxT, NH);

    if (pretrained){
        std::string fp_weights = PREFIX + "bin/gpt2_weights.bin";
        model->load_weights(fp_weights);
    }
}

void setup_tokenizer(Tokenizer*& tk){
    if (tk != nullptr){
        throw std::runtime_error("passed tokenizer pointer must be null");
    }

    std::string fp_tk = PREFIX + "bin/gpt2_vocab.bin";
    tk = new Tokenizer(fp_tk);
}

void setup_datastream(DataStream*& ds, const size_t numtokens){
    if (ds != nullptr){
        throw std::runtime_error("passed datastream pointer must be null");
    }
    std::string fp_ds = PREFIX + "bin/tinyshakespeare.bin";
    ds = new DataStream(fp_ds);
    ds->init_buffer(numtokens + 1); // +1 to include necessary target tokens for training
}

void setup_activations(Activation*& activations,
                       Node*& out,
                       Node*& in, 
                       const size_t B,
                       const size_t T,
                       GPT2*& model){
    activations = new Activation(B, T, model->C, model->L, model->V);
    out = new Node();
    in = new Node();
    activations->point_nodes(out, in);
}



/**
* Allocates and initializes the GPT-2 model, data stream, tokenizer, activations, and nodes for a specified batch size and sequence length.
*
*  @param model A pointer (passed by reference) to a GPT-2 model that will be allocated and initialized.
*  @param ds A pointer (passed by reference) to a DataStream that will be allocated and initialized.
*  @param tk A pointer (passed by reference) to a Tokenizer that will be allocated and initialized.
*  @param activations A pointer (passed by reference) to an Activation object that will be allocated and initialized.
*  @param out A pointer (passed by reference) to an output Node that will be allocated and initialized.
*  @param in A pointer (passed by reference) to an input Node that will be allocated and initialized.
*  @param B The batch size for the model and data stream.
*  @param T The sequence length for the model and data stream.
*  @param pretrained If true, load pre-trained weights for the model.
*
*  @throws `std::invalid_argument` If any of the passed pointers are not nullptr.
*
*  @note Postconditions:
*  @note - The model, data stream, tokenizer, activations, and nodes are allocated and initialized.
*  @note - The model is loaded with pre-trained weights if pretrained is true.
*  @note - The tokenizer is initialized with the vocabulary from GPT-2.
*/

void setup(GPT2*& model,
           DataStream*& ds,
           Tokenizer*& tk,
           Activation*& activations,
           Node*& out,
           Node*& in,
           const size_t B,
           const size_t T,
           const bool pretrained) 
{
    if ((activations != nullptr) || (out != nullptr) || (in != nullptr)){
        throw std::invalid_argument("passed pointers must be empty (nullptr)");
    }

    setup_model(model, pretrained);
    setup_datastream(ds, B*T);
    setup_tokenizer(tk);
    setup_activations(activations, out, in, B, T, model);


    std::cout << "GPT-2 Small" << std::endl;
    std::cout << "max seq len: " << model->maxT << std::endl;
    std::cout << "embedding dimension: " << model->C << std::endl;
    std::cout << "vocab size: " << model->V << std::endl;
    std::cout << "num layers: " << model->L << std::endl;
    std::cout << "num params: " << model->num_params << std::endl;
    std::cout << "allocated batch size: " << B << std::endl;
    std::cout << "allocated sequence length: " << T << std::endl;
    std::cout << "num activations: " << activations->size << std::endl;
    std::cout << "\n";

}




/**
* Deallocates the memory allocated by the setup function.
*
* @param model A pointer (passed by reference) to the GPT-2 model to be deallocated.
* @param ds A pointer (passed by reference) to the DataStream to be deallocated.
* @param tk A pointer (passed by reference) to the Tokenizer to be deallocated.
* @param activations A pointer (passed by reference) to the Activation object to be deallocated.
* @param out A pointer (passed by reference) to the output Node to be deallocated.
* @param in A pointer (passed by reference) to the input Node to be deallocated.
*
* @note Postconditions:
*   @note - All memory allocated by the setup function is released.
*   @note - All passed pointers are set to nullptr.
*/
void tear_down(GPT2*& model,
               DataStream*& ds,
               Tokenizer*& tk,
               Activation*& activations,
               Node*& out,
               Node*& in)
{
    delete out;
    out = nullptr;

    delete in;
    in = nullptr;

    delete activations;
    activations = nullptr;

    delete tk;
    tk = nullptr;

    delete ds;
    ds = nullptr;

    delete model;
    model = nullptr;

    // std::cout << "\nteardown complete, memory deallocated" << std::endl;
}

float mean_loss(Node* loss_node){
    float m_loss = 0.0f;
    size_t numel = loss_node->size;

    for (size_t i = 0; i < numel; i++){
        m_loss += loss_node->act[i];
    }

    m_loss /= numel;

    return m_loss;
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
void train(const int max_batches){
    // setup model
    GPT2* model = nullptr;
    DataStream* ds = nullptr;
    Tokenizer* tk = nullptr;
    Activation* activations = nullptr;
    Node* out = nullptr;
    Node* in = nullptr;
    size_t B = 4;
    size_t T = 64;
    bool pretrained = true;

    setup(model, ds, tk, activations, out, in, B, T, pretrained);

    in->current_T = T; // use all positions for training
    
    // create node for storing losses.
    Node* loss = new Node();
    loss->shape = {B, T};
    loss->size = B*T;
    loss->act = new float[B*T];
    loss->act_grads = new float[B*T];

    // create a node for softmax in
    Node* softmax_in = new Node();
    softmax_in->shape = {B, T, model->V};
    softmax_in->size = B*T*(model->V);
    softmax_in->act = out->act - B*T*(model->V);
    softmax_in->act_grads = out->act_grads - B*T*(model->V);


    // begin training
    for (int t = 0; t < max_batches; t++){
        ds->load_buffer(); // loads B*T + 1 tokens
        ds->buffer_to_Node(in, B*T); // transfer first B*T tokens from buffer to in node
        
        model->clear_kv_cache(); // clear kv cache
        model->forward(out, in); // do forward pass
        crossentropy_forward(loss, out, (ds->buffer) + 1); // compute loss

        float m_loss = mean_loss(loss);
        std::cout << "iteration " << t << ": loss = " << m_loss << std::endl;

        model->zero_grad(); // zero gradients of parameters
        activations->zero_grad(); // zero gradients of activations

        crossentropy_softmax_backward(out, softmax_in, (ds->buffer) + 1, model->softmax->temperature); // do backward through crossentropy loss and softmax
        model->backward(out, in);
        model->update(t+1);
    }


    delete[] loss->act;
    delete[] loss->act_grads;
    delete loss;

    delete softmax_in;

    tear_down(model, ds, tk, activations, out, in);
}


