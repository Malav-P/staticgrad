#include "datastream.hpp"
#include "tokenizer.hpp"
#include "gpt2.hpp"
#include "optimizer.hpp"
#include "interface.hpp"
#include "utils.hpp"
#include <deque>

const std::string PREFIX = REPO_PREFIX;

/**
 * Computes the most probable token at specified position
 *
 * Args:
 *   @param model: Pointer to instance of GPT2
 *   @param tk : Pointer to the tokenizer
 *   @param out: The output node where the computed loss will be stored, a 2D tensor of shape (B, T).
 *   @param in: The input node containing the logits, a 3D tensor of shape (B, T, V).
 *   @param t: Desired position of next token in sequence
 *
 *   @return The most probable token for that position
 */
std::string next_token(GPT2*& model,
                     Tokenizer*& tk,
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

    float* logits = out->act + (t-1)*V;
    uint16_t next_tok = sample_token(logits, V, true);

    in->act[t] = next_tok;

    std::string next_tok_dec = tk->decode({next_tok});

    return next_tok_dec;
}

/**
 * 
 * @brief Helper function for `yap` that prepares the model for generatation, given a piece of input text. 
 *
 * Args:
 *   @param model: Pointer to instance of GPT2
 *   @param tk : Pointer to the tokenizer
 *   @param in: The input node containing the logits, a 3D tensor of shape (B, T, V).
 *   @param start: input string
 *
 *   @return t, the position in the sequence that will be predicted next
 */
size_t prepare_for_gen(GPT2*& model, Tokenizer*& tk, Node*& in, std::string start){
    model->clear_kv_cache();
    size_t T = in->shape[1];
    size_t t;
    uint16_t eot = 50256;
    if (start.empty()){
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

    return t;
}

/**
 * @brief Autoregressive generation of tokens
 *
 * 
 *   @param model: Pointer to instance of GPT2
 *   @param tk   : Pointer to instance of Tokenizer
 *   @param out: The output node where the computed logits are stored, a tensor of shape (B, T, V).
 *   @param in: The input node containing the input tokens, a 2D tensor of shape (B, T).
 *   @param start: Starting sentence given to model
 *
 */
void yap(GPT2*& model,
         Tokenizer*& tk,
         Node*& out,
         Node*& in,
         const std::string start){

    size_t t = prepare_for_gen(model, tk, in, start);

    // autogenerate tokens
    std::cout << start;
    for (size_t i = t; i < in->shape[1]; i++){
        std::string next_tok = next_token(model, tk, out, in, i);
        std::cout << next_tok << std::flush;
    }
}

/**
 * @brief Initializes a GPT-2 model with predefined architecture.
 * 
 * This function allocates and initializes a new GPT-2 model instance with fixed 
 * hyperparameters such as embedding dimension, transformer blocks, vocabulary size, 
 * maximum sequence length, and attention heads. If `pretrained` is set to `true`, 
 * it loads pretrained weights from a binary file.
 * 
 * @param[out] model Pointer to the model instance. Must be `nullptr` before calling.
 * @param[in] pretrained If `true`, loads pretrained weights; otherwise, initializes randomly.
 * 
 * @throws std::runtime_error If `model` is not `nullptr` when passed.
 */
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

/**
 * @brief Initializes the GPT-2 tokenizer.
 * 
 * This function creates a new tokenizer instance using a pre-specified vocabulary file.
 * 
 * @param tk Pointer to the tokenizer instance. Must be `nullptr` before calling.
 * 
 * @throws std::runtime_error If `tk` is not `nullptr` when passed.
 */
void setup_tokenizer(Tokenizer*& tk){
    if (tk != nullptr){
        throw std::runtime_error("passed tokenizer pointer must be null");
    }

    std::string fp_tk = PREFIX + "bin/gpt2_vocab.bin";
    tk = new Tokenizer(fp_tk);
}

/**
 * @brief Initializes a data stream for tokenized text input.
 * 
 * This function creates a new `DataStream` instance from a predefined binary file and 
 * initializes a buffer for token storage.
 * 
 * @param ds Pointer to the data stream instance. Must be `nullptr` before calling.
 * @param numtokens The number of tokens to load into the buffer.
 * 
 * @throws std::runtime_error If `ds` is not `nullptr` when passed.
 * 
 * @note The buffer is initialized with `numtokens + 1` to ensure proper target token availability.
 */
void setup_datastream(DataStream*& ds, const size_t numtokens){
    if (ds != nullptr){
        throw std::runtime_error("passed datastream pointer must be null");
    }
    std::string fp_ds = PREFIX + "bin/tinyshakespeare.bin";
    ds = new DataStream(fp_ds);
    ds->init_buffer(numtokens + 1); // +1 to include necessary target tokens for training
}

/**
 * @brief Initializes activation buffers and associated nodes.
 * 
 * This function allocates memory for activation buffers and initializes input and output nodes
 * for model computations. The activation structure is configured based on the batch size, 
 * sequence length, and model parameters.
 * 
 * @param activations Pointer to the `Activation` instance.
 * @param out Pointer to the output node.
 * @param in Pointer to the input node.
 * @param B Batch size.
 * @param T Sequence length.
 * @param model Pointer to the GPT-2 model instance, which provides channel (`C`), 
 *                  transformer block count (`L`), and vocabulary size (`V`).
 * 
 * @note The `activations` instance internally manages `out` and `in` nodes.
 */
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
 * @brief Initializes the Optimizer.
 * 
 * This function creates a new optimizer instance.
 * 
 * @param opt Pointer to the optimizer instance. Must be `nullptr` before calling.
 * @param model Pointer to a GPT2 instance.alignas
 * @param opt_name the name of the optimizer
 * 
 */
void setup_optimizer(Optimizer*& opt,
                     GPT2*& model,
                     optimizer_t opt_name)
{
    switch (opt_name){
        case ADAM:
            opt = new Adam(model->num_params, model->params, model->grad);
            break;
        default:
            opt = new Adam(model->num_params, model->params, model->grad);
            break;
    }
}


/**
* Allocates and initializes the GPT-2 model, data stream, tokenizer, activations, and nodes for a specified batch size and sequence length.
*
*  @param model A pointer (passed by reference) to a GPT-2 model that will be allocated and initialized.
*  @param opt   A pointer (passed by reference) to an Optimizer instance that will be initialized
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
           Optimizer*& opt,
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
    setup_optimizer(opt, model, ADAM); // hardcoded ADAM optimizer
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
* @param opt A pointer (passed by reference) to an Optimizer instance
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
               Optimizer*& opt,
               DataStream*& ds,
               Tokenizer*& tk,
               Activation*& activations,
               Node*& out,
               Node*& in)
{
    teardown_activations(activations, out, in);
    teardown_tokenizer(tk);
    teardown_datastream(ds);
    teardown_optimizer(opt);
    teardown_model(model);
}

/**
 * @brief Deallocates and resets the Optimizer instance.
 * 
 * 
 * @param opt Pointer to the Optimizer instance. Must be a valid pointer or `nullptr`.
 */
void teardown_optimizer(Optimizer*& opt){
    delete opt;
    opt = nullptr;
}

/**
 * @brief Deallocates and resets the GPT-2 model instance.
 * 
 * This function safely deletes the allocated `GPT2` model and sets the pointer to `nullptr`
 * to prevent dangling references.
 * 
 * @param model Pointer to the model instance. Must be a valid pointer or `nullptr`.
 */
void teardown_model(GPT2*& model){
    delete model;
    model = nullptr;
}

/**
 * @brief Deallocates and resets the data stream instance.
 * 
 * This function safely deletes the allocated `DataStream` instance and sets the pointer 
 * to `nullptr` to avoid memory leaks.
 * 
 * @param ds Pointer to the data stream instance. Must be a valid pointer or `nullptr`.
 */
void teardown_datastream(DataStream*& ds){
    delete ds;
    ds = nullptr;
}

/**
 * @brief Deallocates and resets the tokenizer instance.
 * 
 * This function safely deletes the allocated `Tokenizer` instance and sets the pointer to `nullptr`
 * to prevent unintended access.
 * 
 * @param tk Pointer to the tokenizer instance. Must be a valid pointer or `nullptr`.
 */
void teardown_tokenizer(Tokenizer*& tk){
    delete tk;
    tk = nullptr;
}

/**
 * @brief Deallocates and resets activation buffers and associated nodes.
 * 
 * This function safely deletes the activation buffers along with input and output nodes,
 * setting their pointers to `nullptr` to prevent dangling references.
 * 
 * @param activations Pointer to the `Activation` instance.
 * @param out Pointer to the output node.
 * @param in Pointer to the input node.
 * 
 * @note Ensures all allocated memory for activations and nodes is properly released.
 */
void teardown_activations(Activation*& activations,
                          Node*& out,
                          Node*& in){
    delete out;
    out = nullptr;

    delete in;
    in = nullptr;

    delete activations;
    activations = nullptr;
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
    Optimizer* opt = nullptr;
    DataStream* ds = nullptr;
    Tokenizer* tk = nullptr;
    Activation* activations = nullptr;
    Node* out = nullptr;
    Node* in = nullptr;
    size_t B = 4;
    size_t T = 64;
    bool pretrained = true;

    setup(model, opt, ds, tk, activations, out, in, B, T, pretrained);

    in->current_T = T; // use all positions for training
    
    // create node for storing losses.
    Node* loss = new Node();
    loss->shape = {B, T};
    loss->size = B*T;
    loss->act = new float[B*T];
    loss->act_grads = new float[B*T];

    SoftmaxCrossEntropy* sftmax_ce = new SoftmaxCrossEntropy();



    // begin training
    for (int t = 0; t < max_batches; t++){
        ds->load_buffer(); // loads B*T + 1 tokens
        ds->buffer_to_Node(in, B*T); // transfer first B*T tokens from buffer to in node
        
        model->clear_kv_cache(); // clear kv cache
        model->forward(out, in); // do forward pass
        sftmax_ce->forward(loss, out, (ds->buffer) + 1); // compute loss

        float m_loss = mean_loss(loss);
        std::cout << "iteration " << t << ": loss = " << m_loss << std::endl;

        opt->zero_grad(); // zero gradients of parameters
        activations->zero_grad(); // zero gradients of activations

        sftmax_ce->backward(loss, out, (ds->buffer) + 1); // do backward through crossentropy loss and softmax
        model->backward(out, in);
        opt->update();
    }

    delete sftmax_ce;

    delete[] loss->act;
    delete[] loss->act_grads;
    delete loss;

    tear_down(model, opt, ds, tk, activations, out, in);
}


