#include "classes.hpp"
#include <cmath>
#include <cfloat>
#include <cstring>
#include <stdexcept>
#include <iostream>

#ifdef APPLE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif


/**
 * @brief Shifts the pointer location for in and out activations/grad in order to process the next layer in gpt. Used for the forward pass
 *
 *   @param out: The output node where the next layer's output will be written to
 *   @param in: The input node containing the input activations for the next layer
 *   @param shape_: The shape of the next layer's output
 *
 */
void shift(Node* out, Node* in, const std::vector<size_t> shape_){

    in->act = out->act;
    in->act_grads = out->act_grads;
    in->shape = out->shape;
    in->size = out->size;

    out->act += out->size;
    out->act_grads += out->size;
    
    out->shape = shape_;

    size_t numel = 1;

    for (size_t element : shape_) {
        numel *= element;
    }

    out->size = numel;
}

/**
 * @brief Shifts the pointer location for in and out activations/grad in order to process the previous layer in gpt (used for the backward pass)
 *
 * @param out: The output node where the next layer's output will be written to
 * @param in: The input node containing the input activations for the next layer
 * @param shape_: The shape of the previous layer's input
 *
 */
void shift_back(Node* out, Node* in, const std::vector<size_t> shape_){

    out->act = in->act;
    out->act_grads = in->act_grads;
    out->shape = in->shape;
    out->size = in->size;

    in->shape = shape_;

    size_t numel = 1;

    for (size_t element : shape_) {
        numel *= element;
    }

    in->size = numel;
    in->act -= in->size;
    in->act_grads -= in->size;

}

/**
 * @brief Embeds the input sequence into latent space using token embedding and position embedding weight matrices.
 *
 * @param out The output node containing the encoded representation.
 * @param in The input node containing the sequence to be encoded.
 *
 * @note The input node's shape should be [B, T], where B is the batch size and T is the sequence length.
 * The output node's shape will be [B, T, C], where C is the embedding size.
 *
 * @note
 * This function applies the embedding lookup for both tokens and positions and then sums them up to get the final encoded representation.
 * The token embeddings are stored in the 'params' array, with the first C * vocab_size elements representing the token embeddings.
 * The position embeddings are stored in the rest of the 'params' array, with the next C * maxT elements representing the position embeddings.
 * The embedded representation is stored in the 'act' member of the out node.
 */
void Encoder::forward(Node* out, Node* in){

    size_t B = in->shape[0];
    size_t T = in->shape[1];

    size_t current_T = in->current_T;


    float* wte = params;
    float* wpe = params + C * vocab_size;

    for (size_t b = 0; b < B; b++){
        for (size_t t = 0; t < current_T; t++){
            // get token_id (lrint casts the double to a long int)
            long int token_id = lrint(in->act[b*T + t]);
            
            // seek to token embedding, wte is (V, C)
            float* token_embed = wte + token_id*C;

            // seek to position emebedding, wpe is (maxT, C)
            float* pos_embed = wpe + t*C;

            // seek to output position
            float* output = out->act + b*T*C + t*C;

            for (size_t i = 0; i < C; i++){
                output[i] = token_embed[i] + pos_embed[i];
            }
        }
    }
}

/**
 * @brief Performs the backward pass through the embedding layer.
 *
 * This function updates the gradient of token embeddings (wte) and positional embeddings (wpe) based on the gradients of the output.
 *
 * @param out The output tensor of shape (B, T, C). B is the batch size, T is the sequence length, and C is the hidden channel size.
 * @param in The input tensor of shape (B, T). Each element in the input tensor corresponds to an integer representing the token ID.
 *
 * @note 
 * - The function assumes that the gradient buffer (grad) of the Encoder class is properly allocated and has enough space to accommodate the gradients of wte and wpe.
 * The gradients of wte are stored in the first part of the grad buffer, and the gradients of wpe are stored in the second part, following the layout: [wte_gradients, wpe_gradients].
 * 
 * @note
 * - The input `in` is expected to contain token indices (floating-point values) that are
 *       subsequently cast to `long int` to access the corresponding embeddings.
 */
void Encoder::backward(Node* out, Node* in){

    // out is (B, T, C)
    // wte is (V, C)
    // wpe is (maxT, C)

    size_t B = in->shape[0];
    size_t T = in->shape[1];

    float* wte_g = grad;
    float* wpe_g = grad + C * vocab_size;

    for (size_t b = 0; b < B; b++){
        for (size_t t = 0; t < T; t++){
            // get token_id (lrint casts the double to a long int)
            long int token_id = lrint(in->act[b*T + t]);
            
            // seek to token embedding grad, wte_g is (V, C)
            float* token_embed_g = wte_g + token_id*C;

            // seek to position emebedding, wpe_g is (maxT, C)
            float* pos_embed_g = wpe_g + t*C;

            // seek to output position
            float* output_g = out->act_grads + b*T*C + t*C;

            for (size_t i = 0; i < C; i++){
                float deriv = output_g[i];
                token_embed_g[i] += deriv;
                pos_embed_g[i] += deriv;
            }
        }
    }
    
}

/**
 * @brief Constructor for the TransformerBlock class.
 *
 * This constructor initializes a Transformer block layer.
 * It comprises multiple layers such as LayerNorm, Multi-Head Self-Attention, Feed-Forward Neural Network (FFN), and residual connections.
 *
 * @param params_ Pointer to the parameters of the Transformer block. The parameters should be organized in a specific order as mentioned in the code.
 * @param grad_   Pointer to the gradients of the Transformer block. The gradients should be organized in the same order as the parameters.
 * @param C       The number of channels or embedding dimension in the input data.
 * @param NH      The number of heads for the multi-head self-attention. This parameter determines the parallelism of the self-attention mechanism.
 * @param maxT    The maximum sequence length. This is important for allocating sufficient memory and handling variable-length sequences.
 *
 * @throw `std::invalid_argument` If @p C is not divisible by @p NH. This is a fundamental requirement of the Transformer architecture.
 * 
 * @note 
 * - The constructor dynamically allocates memory for various internal nodes and layers. Hence, it is crucial to handle exceptions during construction.
 */
TransformerBlock::TransformerBlock(float* params_, float* grad_, const size_t C, const size_t NH):
    Operation(params_, grad_),
    res1_node(nullptr),
    res2_node(nullptr) {

        if (C % NH != 0){
            throw std::invalid_argument("C must be divisible by NH");
        }

        res1_node = new Node();
        res2_node = new Node();

        float* layer_param = params_;
        float* layer_grad = grad_;


        ln1 = new LayerNorm(layer_param, layer_grad);
        layer_param += C + C; // C weights and C biases
        layer_grad += C + C;


        mat1 = new Matmul(layer_param, layer_grad);
        layer_param += C * 3*C + 3*C;
        layer_grad +=  C * 3*C + 3*C;

        att = new Attention(NH);

        mat2 = new Matmul(layer_param, layer_grad);
        layer_param += C*C + C;
        layer_grad +=  C*C + C;

        res1 = new Add();

        ln2 = new LayerNorm(layer_param, layer_grad);
        layer_param += C + C;
        layer_grad += C + C;

        mat3 = new Matmul(layer_param, layer_grad);
        layer_param += C * 4*C + 4*C;
        layer_grad += C * 4*C + 4*C;

        gelu = new GELU();

        mat4 = new Matmul(layer_param, layer_grad);
        layer_param += 4*C * C + C;
        layer_grad += 4*C * C + C;

        res2 = new Add();
}


/**
 * @brief Performs a forward pass through the transformer block.
 *
 * @param[out] out  The output node.
 * @param[in]  in   The input node with shape `(B, T, C)`.
 *
 * @throw std::runtime_error if the output node and the internal output node do not match.
 * This indicates an incorrect implementation of activation storage.
 */
void TransformerBlock::forward(Node* out, Node* in){   // (B, T, C) -> (B, T, C)
    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t C = in->shape[2];

    res1_node->act = in->act;
    res1_node->act_grads = in->act_grads;
    res1_node->shape = in->shape;
    res1_node->size = in->size;

    Node* input = new Node();
    input->act = in->act;
    input->act_grads = in->act_grads;
    input->shape = in->shape;
    input->size = in->size;
    input->current_T = in->current_T;


    Node* output = new Node();
    output->act = input->act + B * T * C;
    output->act_grads = input->act_grads + B * T * C;
    output->shape = {B, T, C};
    output->size = B * T * C;
    output->current_T = in->current_T;

    // first layer norm
    ln1->forward(output, input);

    // first matmul and bias
    shift(output, input, {B, T, 3*C});
    mat1->forward(output, input);
    
    // attention
    shift(output, input, {B, T, C});
    att->forward(output, input);

    // second matmul and bias
    shift(output, input, {B, T, C});
    mat2->forward(output, input);

    // residual
    res1->forward(output, res1_node); // in place

    // store activations for second residual
    res2_node->act = output->act;
    res2_node->act_grads = output->act_grads;
    res2_node->shape = output->shape;
    res2_node->size = output->size;

    // second layer norm
    shift(output, input, {B, T, C});
    ln2->forward(output, input);

    // third matmul and bias
    shift(output, input, {B, T, 4*C});
    mat3->forward(output, input); // (B, T, C) -> (B, T, 4*C)

    // GELU
    shift(output, input, {B, T, 4*C}); 
    gelu->forward(output, input); // (B, T, 4C) - > (B, T, 4C)

    // fourth matmul and bias
    shift(output, input, {B, T, C});
    mat4->forward(output, input);

    // second residual
    res2->forward(output, res2_node); // in place (B, T, C) -> (B, T, C)


    // verify that results are in out Node
    if ((output->act != out->act) || (output->act_grads != out->act_grads) || (output->size != out->size)){
        throw std::runtime_error("out node and output node are not equal in transformer block");
    }

    // free memory of helper nodes
    delete input;
    delete output;

}

/**
 * @brief Performs a backward pass through the transformer block.
 *
 * @param[out] out  The output node.
 * @param[in]  in   The input node with shape `(B, T, C)`.
 *
 * @throw std::runtime_error if the input node and the internal input node do not match.
 * This indicates an incorrect implementation of activation storage.
 */
void TransformerBlock::backward(Node* out, Node* in){

    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t C = in->shape[2];

    Node* output = new Node();
    output->act = out->act;
    output->act_grads = out->act_grads;
    output->shape = out->shape;
    output->size = out->size;

    Node* input = new Node();
    input->act = out->act - B * T * 4*C;
    input->act_grads = out->act_grads - B * T * 4*C;
    input->shape = {B, T, 4*C};
    input->size = B * T * 4*C;

    // backward through second residual
    res2->backward(output, res2_node);

    // backward through fourth matmul and bias
    mat4->backward(output, input);

    // backward through GELU
    shift_back(output, input, {B, T, 4*C}); // (B, T, 4*C) is input shape of GELU operation
    gelu->backward(output, input);

    // backward through third matmul and bias
    shift_back(output, input, {B, T, C}); // (B, T, C) is input shape of matmul operation
    mat3->backward(output, input);

    // backward through second layernorm
    shift_back(output, input, {B, T, C}); // (B, T, C) is the input shape of layernorm operation
    ln2->backward(output, input);

    // backward through first residual
    shift_back(output, input, {B, T, C}); // (B, T, C) is input shape of res1
    res1->backward(output, res1_node);

    // backward through second matmul and bias
    mat2->backward(output, input);

    // backward through attention
    shift_back(output, input, {B, T, 3*C}); // (B, T, 3*C) is input shape of attention
    att->backward(output, input);

    // backward through first matmul and bias
    shift_back(output, input, {B, T, C}); // (B, T, C) is input shape of matmul operatnion
    mat1->backward(output, input);

    // backward through first layernorm
    shift_back(output, input, {B, T, C}); // (B, T, C) is input shape of layernorm
    ln1->backward(output, input);

    // verify that results are in 'in' Node
    if ((input->act != in->act) || (input->act_grads != in->act_grads) || (input->size != in->size)){
        std::string errorMessage = "in node and input node are not equal. Difference of pointer locations: " + std::to_string(input->act - in->act);
        throw std::runtime_error(errorMessage);
        
    }

    // free allocated memory for helper nodes
    delete output;
    delete input;

}

/**
 * @brief Destructor for transformer block. Frees resources associated with block.
 *
 */
TransformerBlock::~TransformerBlock(){
    delete res2;
    delete mat4;
    delete gelu;
    delete mat3;
    delete ln2;
    delete res1;
    delete mat2;
    delete att;
    delete mat1;
    delete ln1;

    delete res1_node;
    delete res2_node;


}

/**
 * @brief Performs a forward pass through the attention block.
 *
 * @param out  The output node.
 * @param  in   The input node with shape `(B, T, C)`.
 *
 * @note 
 * - memory for an internal buffer is allocated to cache the values of the query-key dot products and the softmax operation on that cache.
 * @note 
 * - any resources stored in the buffer are freed before the above allocation is done.
 */
void Attention::forward(Node* out, Node* in){ // (B, T, 3C) -> (B, T, C)

    if (in->size != 3 * out->size){
        throw std::invalid_argument("input must be of shape (B, T, 3C) and output must be of shape (B, T, C)");
    }

    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t C = in->shape[2] / 3;

    size_t current_T = in->current_T;
    size_t head_size = C / num_heads;

    int half_buffer_size = num_heads*B*lrint(T*(T+1)/2);

    size_t start;
    if (buffer != nullptr){ // kv cache exists
        start = current_T - 1;
    }
    else { // kv cache does not exist (this should always be the case during training)
        start = 0;
        buffer = new float[2*half_buffer_size];
    }    
    
    for (size_t b = 0 ; b < B; b++){

        for (size_t t = start; t < current_T; t++){

            for (size_t h = 0 ; h < num_heads; h++){

                // get presoftmax buffer at b,t,h position
                float* pre_softmax_bth = buffer + b*num_heads*lrint(T*(T+1)/2) + lrint(t*(t+1)/2)*num_heads + (t + 1)*h;

                // get query vector
                float* q = in->act + b * T * 3*C + t * 3*C + h * head_size;

                // get raw scores
                float maxval = -FLT_MAX;
                for (size_t t2 = 0; t2 <= t; t2++){
                    // get key vector for t2 token
                    float* k_t2 = in->act + b * T * 3*C + t2 * 3*C + C + h * head_size;

                    // dot key vector with query vector to get score
                    float score = 0.0f;
                    for (size_t i = 0; i < head_size; i++){
                        score += q[i] * k_t2[i];
                    }

                    score = score / sqrtf(head_size);

                    if (score > maxval){
                        maxval = score;
                    }   

                    pre_softmax_bth[t2] = score;       

                }

                // get post softmax buffer at b,t,h position
                float* post_softmax_bth = buffer + half_buffer_size + b*num_heads*lrint(T*(T+1)/2) + lrint(t*(t+1)/2)*num_heads + (t + 1)*h;

                // normalize scores according to softmax
                float exp_sum = 0.0f;
                for (size_t t2 = 0; t2 <= t; t2++){
                    float exp_score = expf(pre_softmax_bth[t2] - maxval);
                    post_softmax_bth[t2] = exp_score;
                    exp_sum += exp_score;
                }

                float exp_sum_inv = exp_sum == 0 ? 0.0f : 1.0f / exp_sum;

                for (size_t t2 = 0; t2 <= t; t2++){
                    post_softmax_bth[t2] *= exp_sum_inv;
                }


                // get output vector
                float* out_act = out->act + b * T * C + t * C + h * head_size;
                // zero the output
                for (size_t i = 0; i < head_size; i++){
                    out_act[i] = 0.0f;
                }

                // accumulate weighted scores from tokens
                for (size_t t2 = 0; t2 <= t; t2++){
                    // find value vector for t2 token
                    float* v_t2 = in->act + b * T * 3*C + t2 * 3*C + C + C + h * head_size;

                    for (size_t i=0; i < head_size; i++){
                        out_act[i] += post_softmax_bth[t2] * v_t2[i];
                    }
                }
            }
        }
    }
}

/**
 * @brief Performs a backward pass through the attention block.
 *
 * @param[out] out  The output node.
 * @param[in]  in   The input node with shape `(B, T, C)`.
 *
 * @note 
 * - memory for an internal buffer is allocated to cache the values of the gradients for query-key dot products and the gradients of softmax operation on that cache.
 * @note 
 * - any resources stored in the buffer are freed before the above allocation is done.
 * 
 * @throw `std::runtime_error` if `buffer` from forward pass has not been allocated (and thus written to).
 */
void Attention::backward(Node* out, Node* in) {
    // Get the batch size, sequence length, and number of channels
    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t C = in->shape[2] / 3;

    // Calculate the size of each attention head
    size_t head_size = C / num_heads;

    // Check if the buffer has been allocated properly
    if (buffer == nullptr){
        std::__throw_runtime_error("buffer was not allocated properly. This may be because backward() was called before forward()");
    }

    // Allocate memory for the gradients of the buffer
    int half_buffer_size = num_heads*B*lrint(T*(T+1)/2);
    delete[] dbuffer;
    dbuffer = new float[2*half_buffer_size]{0};

    // Iterate over each batch
    for (size_t b = 0 ; b < B; b++){
        // Iterate over each token
        for (size_t t = 0; t < T; t++){
            // Iterate over each attention head
            for (size_t nh = 0 ; nh < num_heads; nh++){

                // Get the output gradient
                float* out_g = out->act_grads + b * T * C + t * C + nh * head_size;

                // Get the softmax output and gradients of the softmax output
                float* post_softmax = buffer + half_buffer_size + b*num_heads*lrint(T*(T+1)/2) + lrint(t*(t+1)/2)*num_heads + (t + 1)*nh;
                float* dpost_softmax = dbuffer + half_buffer_size + b*num_heads*lrint(T*(T+1)/2) + lrint(t*(t+1)/2)*num_heads + (t + 1)*nh;

                // Backward pass through the accumulation of the softmax output and value vector
                for (size_t t2 = 0; t2 <= t; t2++){
                    // Get value vector and its gradient for t2-th token
                    float* dv_t2 = in->act_grads + b * T * 3*C + t2 * 3*C + C + C + nh * head_size;
                    float* v_t2  = in->act       + b * T * 3*C + t2 * 3*C + C + C + nh * head_size;

                    // Calculate the gradient of post_softmax and the value vector
                    for (size_t i=0; i < head_size; i++){
                        dpost_softmax[t2] += out_g[i] * v_t2[i];
                        dv_t2[t2] += out_g[i] * post_softmax[t2];
                    }
                }

                // get dbuffer1
                float* dpre_softmax = dbuffer + b*num_heads*lrint(T*(T+1)/2) + lrint(t*(t+1)/2)*num_heads + (t + 1)*nh;

                // backward through softmax post_softmax = softmax(buffer1)
                for (size_t t2 = 0; t2 <= t; t2++){
                    for (size_t t3 = 0; t3 <= t; t3++){
                        float local_deriv = post_softmax[t3] * ((t2 == t3 ? 1.0f : 0.0f) - post_softmax[t2] );
                        dpre_softmax[t2] += dpost_softmax[t3] * local_deriv; 
                    }
                    
                }


                // backward through query key dot products
                // get query vector and its grad
                float* q = in->act + b * T * 3*C + t * 3*C + nh * head_size;
                float* dq = in->act_grads + b * T * 3*C + t * 3*C + nh * head_size;
                for (size_t t2 = 0; t2 <= t; t2++){
                    // find key vector for t2 token
                    float* k_t2 = in->act + b * T * 3*C + t2 * 3*C + C + nh * head_size;
                    float* dk_t2 = in->act_grads + b * T * 3*C + t2 * 3*C + C + nh * head_size;

                    for (size_t i = 0; i < head_size; i++){
                        dq[i] += dpre_softmax[t2] * k_t2[i] / sqrtf(head_size);
                        dk_t2[i] += dpre_softmax[t2] * q[i] / sqrtf(head_size);
                    }
                    
                }
            }

        }

    }
}

/**
 * @brief Forward pass of the GELU (Gaussian Error Linear Unit) activation function.
 *
 * This function applies the GELU activation function to each element in the input node and stores the result in the output node.
 *
 * The GELU activation function is defined as:
 * gelu(x) = 0.5 * x * (1 + erf(x/sqrt(2)))
 * where erf is the error function, also known as the Gauss error function.
 * However, the error function can be approximated using the hyperbolic tangent function (tanh) as follows:
 * erf(x) ≈ tanh(sqrt(2/π) * (x + 0.044715 * x^3))
 * This approximation is used in this function to compute the GELU activation function.
 *
 * @param out Pointer to the output node.
 * @param in Pointer to the input node.
 */
void GELU::forward(Node* out, Node* in) {
    size_t numel = in->size;

    float x;

    for (size_t i = 0; i < numel; i++){
        x = in->act[i];
        out->act[i] = 0.5f * x * (1 + tanhf( sqrtf(2.0f/M_PI) * (x + 0.044715f * x * x * x)));
    }
}

/**
 * @brief Backward pass of the GELU (Gaussian Error Linear Unit) activation function.
 *
 * This function applies the gradient of GELU activation function to each element.
 *
 * @see `GELU::forward` for details of GELU
 * 
 * @param out Pointer to the output node.
 * @param in Pointer to the input node.
 */
void GELU::backward(Node* out, Node* in) {
    size_t numel = in->size;

    float x;

    float tanh_inp;
    float tanh_out;
    float sech_out;

    for (size_t i = 0; i < numel; i++){
        x = in->act[i];
        tanh_inp = sqrtf(2.0f/M_PI) * (x + 0.044715f * x * x * x);
        tanh_out = tanhf(tanh_inp);
        sech_out = 1.0f / coshf(tanh_inp);

        in->act_grads[i] += out->act_grads[i] * ( 0.5f * (1.0f + tanh_out) + 0.5f * x * (sech_out * sech_out * sqrtf(2.0f/M_PI) * (1.0f + 3.0f* 0.044715f * x * x)));
    }
}



/**
 * @brief Forward pass of the Layernorm operation
 *
 * This function normalizes each row of the input (along axis = 2) to have zero mean and unit variance. Then 
 * it applies a scale and shift to each element. There are a total of 2 * C learnable parameters in the layernorm.
 *
 * 
 * @param out Pointer to the output node.
 * @param in Pointer to the input node.
 */
void LayerNorm::forward(Node* out, Node* in) { // (B, T, C) -> (B, T, C)


    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t C = in->shape[2];

    size_t current_T = in->current_T;


    delete[] rstd; delete[] m;
    rstd = new float[B*T];
    m = new float[B*T];


    float eps = 1e-5; // division by zero prevention during normalization


    for (size_t b = 0; b < B; b++){
        for (size_t t = 0; t < current_T; t++){
            float* inp = in->act + b*T*C + t*C;
            float m_ = 0.0f;

            // calculate mean
            for (size_t i = 0; i < C; i++){
                m_ += inp[i];
            }
            m_ = m_/C;

            // calculate var
            float v = 0.0f;
            for (size_t i = 0 ; i < C; i++){
                float dev = inp[i] - m_;
                v += dev*dev;
            }
            v = v/C;

            if (std::abs(v) < eps) {
                std::cerr << "Variance in layernorm close to zero." << std::endl;
            }

            // write to output
            float* out_ = out->act + b*T*C + t*C;
            float rstd_ = 1.0f / sqrtf(v+ eps);
            float* w_ = params;
            float* b_ = params+C;

            for (size_t i = 0; i < C; i++){
                float norm_inp = rstd_* (inp[i] - m_);
                out_[i] = norm_inp * w_[i] + b_[i];
            }

            // store rstd_ and m_ for backward pass
            rstd[b * T + t] = rstd_;
            m[b * T + t] = m_;

        }
    }
}

/**
 * @brief Backward pass of the Layernorm operation
 * 
 * This function applies the backward pass of thel layernorm operation.
 *
 * @see `LayerNorm::forward` for details about the operation
 * 
 * @param out Pointer to the output node.
 * @param in Pointer to the input node.
 */
void LayerNorm::backward(Node* out, Node* in){


    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t C = in->shape[2];


    float* w_g = grad;
    float* b_g = grad+C;

    float* w = params;

    for (size_t b = 0; b < B; b++){
        for (size_t t = 0; t < T; t++){
            float* out_g = out->act_grads + b*T*C + t*C;
            float* in_g = in->act_grads + b*T*C + t*C;
            float* inp = in->act + b*T*C + t*C;


            float m_ = m[b*T + t];
            float rstd_ = rstd[b*T + t];

            // first pass to store some useful values.
            float norm_g_mean = 0.0f;
            float norm_g_mean_times_norm = 0.0f;
            for (size_t i = 0; i < C; i++){
                float n = rstd_*(inp[i] - m_);
                float dnorm = out_g[i] * w[i];

                norm_g_mean += dnorm;

                norm_g_mean_times_norm += dnorm * n;
            }

            norm_g_mean = 1.0f/C * norm_g_mean;
            norm_g_mean_times_norm = 1.0f/C * norm_g_mean_times_norm;

            for (size_t i = 0; i < C; i++){
                // bias gradient
                b_g[i] += out_g[i];
                
                // scale gradient
                w_g[i] += (inp[i] - m_)*rstd_ * out_g[i];

                // input gradient
                float dnorm = out_g[i] * w[i];
                float n = rstd_*(inp[i] - m_);
                in_g[i] += rstd_ * (dnorm - norm_g_mean - norm_g_mean_times_norm * n);
            }

        }
    }
}

/**
 * @brief Computes the forward pass of matrix multiplication.
 *
 * This method performs a matrix multiplication operation on the input node's
 * activation values and the layer's parameters, storing the result in the output
 * node's activation values.
 *
 * @param out The output node where the result of the matrix multiplication will be stored.
 * @param in The input node containing the activation values to be multiplied.
 *
 * @throws `std::invalid_argument` If either the input or output node's activation values are nullptr.
 */    
void Matmul::forward(Node* out, Node* in) {

    if ((in->act == nullptr) || (out->act == nullptr)){
        throw std::invalid_argument("in or out activations are nullptrs.");
    }

    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t C = in->shape[2];
    size_t OC = out->shape[2];

    size_t current_T = in->current_T;
    
    float alpha = multiplier, beta = 0.0f;
    float* B_ = params;

    int M = current_T; // rows of A and C
    int N = OC; // columns of B and C
    int K = C; // columns of A and rows of B

    int lda = C; // leading dimension of A
    int ldb = OC; // leading dimension of B
    int ldc = OC; // leading dimension of C


    for (size_t b = 0; b < B; b++){

        float* A = in->act + b * T * C;
        float* out_ = out->act + b * (T) * (OC);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, alpha, A, lda, B_, ldb, beta, out_, ldc);
        
        if (bias){
            float* param_b = params + C*OC;
            for (size_t t = 0; t < current_T; t++){
                for (size_t c = 0; c < OC; c++){
                    out_[t*OC + c] += param_b[c];
                }
            }
        }

    }

}




/**
 * @brief Computes the forward pass of matrix multiplication assuming the weights are given tranposed.
 *
 * This method performs a matrix multiplication operation on the input node's
 * activation values and the layer's parameters, storing the result in the output
 * node's activation values. in has shape (B, T, C).  params has shape (OC, C). out has shape (B, T, OC).
 *
 * @param out The output node where the result of the matrix multiplication will be stored.
 * @param in The input node containing the activation values to be multiplied.
 *
 * @throws std::invalid_argument If either the input or output node's activation values are nullptr.
 */    
void Matmul::forward2(Node* out, Node* in) {

    if ((in->act == nullptr) || (out->act == nullptr)){
        throw std::invalid_argument("in or out activations are nullptrs.");
    }

    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t C = in->shape[2];
    size_t OC = out->shape[2];

    size_t current_T = in->current_T;

    
    float alpha = multiplier, beta = 0.0f;
    float* B_ = params;

    int M = current_T; // rows of A and C
    int N = OC; // columns of trans(B) and C
    int K = C; // columns of A and rows of B

    int lda = C; // leading dimension of A
    int ldb = C; // leading dimension of B
    int ldc = OC; // leading dimension of C


    for (size_t b = 0; b < B; b++){

        float* A = in->act + b * T * C;
        float* C_ = out->act + b * (T) * (OC);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, N, K, alpha, A, lda, B_, ldb, beta, C_, ldc);

        if (bias){
            float* param_b = params + C*OC;
            for (size_t t = 0; t < current_T; t++){
                for (size_t c = 0; c < OC; c++){
                    C_[t*OC + c] += param_b[c];
                }
            }
        }

    }
}

/**
 * @brief Computes the backward pass of matrix multiplication.
 *
 * This method performs a matrix multiplication operation to compute gradients of with respect to
 * the input and parameters
 *
 * @param out The output node where the result of the matrix multiplication will be stored.
 * @param in The input node containing the activation values to be multiplied.
 *
 * @throws `std::invalid_argument` If either the input or output node's activation values are nullptr.
 */  
void Matmul::backward(Node* out, Node* in) {
    // Compute C <- alpha * op(A) * op(B) + beta*C
    // where op() can be the identity operation or the transpose operation.
    size_t B = in->shape[0];
    size_t m = in->shape[1], k = in->shape[2];
    size_t p = k, n = out->shape[2];
    float alpha = multiplier, beta = 1.0f;

    float* B_ = params;
    float* dLdB = grad;


    for (size_t b = 0; b < B; b++){

        float* A = in->act + b * m * k;
        float* dLdA = in->act_grads + b * m * k;
        float* dLdC = out->act_grads + b * m * n;

        // gradient wrt bias
        if (bias){
            float* grad_b = grad + k*n;
            for (size_t t = 0; t < m; t++){
                for (size_t c = 0; c < n; c++){
                    grad_b[c] += dLdC[t*n + c];
                }
            }
        }

        // gradient wrt input
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    m, k, n, alpha, dLdC, n, B_, n, beta, dLdA, k);
        // gradient wrt weights
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    p, n, m, alpha, A, k, dLdC, n, beta, dLdB, n);

    }
}

/**
 * @brief Computes the backward pass of matrix multiplication assuming the weights are given tranposed.
 *
 * This method performs a matrix multiplication operation to compute gradients of with respect to
 * the input and parameters
 * 
 * @param out The output node where the result of the matrix multiplication will be stored.
 * @param in The input node containing the activation values to be multiplied.
 *
 * @throws std::invalid_argument If either the input or output node's activation values are nullptr.
 */  
void Matmul::backward2(Node* out, Node* in) {
    // Compute C <- alpha * op(A) * op(B) + beta*C
    // where op() can be the identity operation or the transpose operation.
    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t C = in->shape[2];
    size_t OC = out->shape[2];


    float alpha = multiplier, beta = 1.0f;

    int M, N, K, lda, ldb, ldc;

    float* B_ = params;
    float* dLdB = grad;


    for (size_t b = 0; b < B; b++){

        float* A = in->act + b * T * C;
        float* dLdA = in->act_grads + b * T * C;
        float* dLdC = out->act_grads + b * T * OC;

        // gradient wrt bias
        if (bias){
            float* grad_b = grad + OC*C;
            for (size_t t = 0; t < T; t++){
                for (size_t c = 0; c < OC; c++){
                    grad_b[c] += dLdC[t*OC + c];
                }
            }
        }

        // gradient wrt input
        M = T; // rows of dL/dC and dL/dA
        N = C; // columns of B and dL/dA
        K = OC; // columns of dL/dC and rows of B
        lda = OC; // leading dimension of dL/dC
        ldb = C; // leading dimension of B
        ldc = C; // leading dimension of dL/dA
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, alpha, dLdC, lda, B_, ldb, beta, dLdA, ldc);

        // gradient wrt parameters
        M = OC; // rows of dL/dC^T and dL/dB
        N = C; // columns of A and dL/dB
        K = T; // rows of dL/dC and A
        lda = OC; // leading dimension of dL/dC
        ldb = C; // leading dimension of A
        ldc = C; // leading dimension of dL/dB
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    M, N, K, alpha, dLdC, lda, A, ldb, beta, dLdB, ldc);

    }
}

/**
 * @brief Performs element-wise addition of two nodes.
 * 
 * This function adds the activation values (`act`) of the input node (`in`)
 * to the output node (`out`). The sizes of the input and output arrays must be equal.
 * 
 * @param[out] out The output node where the result is stored. Must have a valid `act` pointer.
 * @param[in] in The input node whose activation values will be added. Must have a valid `act` pointer.
 * 
 * @throws `std::invalid_argument` If either `out->act` or `in->act` is null.
 * @throws `std::invalid_argument` If `out->size` is not equal to `in->size`.
 * 
 * @note This function modifies `out->act` in place.
 */
void Add::forward(Node* out, Node* in){

    if (out->act == nullptr || in->act == nullptr){
        throw std::invalid_argument("pointer to data is null");
    }

    if (out->size != in->size){
        throw std::invalid_argument("sizes of input and output arrays unequal");
    }

    size_t numel = in->size;

    for (size_t i = 0; i < numel; i++){
        out->act[i] += in->act[i];
    }

}

/**
 * @brief Performs backwards pass of `Add` operation.
 * 
 * 
 * @param[out] out The output node where the result is stored. Must have a valid `act` pointer.
 * @param[in] in The input node whose activation values will be added. Must have a valid `act` pointer.
 * 
 * @throws `std::invalid_argument` If either `out->act` or `in->act` is null.
 * @throws `std::invalid_argument` If `out->size` is not equal to `in->size`.
 * 
 * @note This function modifies `in->act` in place.
 */
void Add::backward(Node* out, Node* in){

    if (out->act == nullptr || in->act == nullptr){
        throw std::invalid_argument("pointer to data is null");
    }

    if (out->size != in->size){
        throw std::invalid_argument("sizes of input and output arrays unequal");
    }

    size_t numel = in->size;

    for (size_t i = 0; i < numel ; i++){
        in->act_grads[i] += out->act_grads[i];
    }
}

/**
 * @brief Performs row-wise addition on the input tensor.
 * 
 * This function adds the parameter vector (`params`) to each row of the input tensor (`in`),
 * storing the result in the output tensor (`out`). The addition is performed along the 
 * last dimension (channel dimension).
 * 
 * @param[out] out The output node storing the modified activations. 
 *                 Must have a valid `act` pointer and match `in` in shape.
 * @param[in] in The input node whose activations are modified. 
 *               Must have a valid `act` pointer.
 * 
 * @note The input tensor has a shape of `[B, T, C]`, where:
 *       - `B` = batch size
 *       - `T` = sequence length (time steps)
 *       - `C` = number of channels (features per timestep)
 * 
 * @details The operation is performed as:
 *          \f$ out[b, t, c] = in[b, t, c] + params[c] \f$
 *          for all `b` in `B`, `t` in `current_T`, and `c` in `C`.
 */
void RowAdd::forward(Node* out, Node* in){

    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t C = in->shape[2];

    size_t current_T = in->current_T;


    for (size_t b = 0; b < B; b++){

        float* out_ = out->act + b * T * C;
        float* inp0 = in->act + b * T * C;

        for (size_t t = 0; t < current_T; t++){
            for (size_t c = 0; c < C; c++){
                out_[t*C + c] = inp0[t*C + c] + params[c];
            }
        }
    }
}

/**
 * @brief Computes the gradients for the row-wise addition operation.
 * 
 * This function propagates gradients back through the computation graph. It 
 * computes the gradients for both the input tensor (`in`) and the parameter vector (`params`).
 * 
 * @param[out] out The output node containing gradient values (`act_grads`). 
 * @param[in] in The input node whose gradient will be updated (`act_grads`). 
 * 
 * @note This function updates:
 *       - `in->act_grads` (gradients for the input tensor).
 *       - `grad` (gradients for the parameter vector).
 * 
 * @details The gradients are computed as:
 *          \f$ \text{inp1_g}[c] += \sum_{b=0}^{B} \sum_{t=0}^{T} \text{out_g}[b, t, c] \f$
 *          ensuring correct gradient accumulation for the parameter vector.
 */
void RowAdd::backward(Node* out, Node* in){

    float* inp0_g = in->act_grads;
    float* inp1_g = grad;
    float* out_g = out->act_grads;

    // backward pass for first input
    if (out_g != inp0_g){
        memcpy(inp0_g, out_g, out->size * sizeof(float));
    }

    // backward pass for second input
    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t C = in->shape[2];

    for (size_t b = 0; b < B; b++){    
        for (size_t t = 0; t < T; t++){
            for (size_t c = 0; c < C; c++){
                inp1_g[c] += out_g[b*T*C + t*C + c];
            }
        }
    }
}