#include "classes.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

#ifdef APPLE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif


/**
 * Shifts the pointer location for in and out activations/grad in order to process the next layer in gpt. Used for the forward pass
 *
 * Args:
 *   @param out: The output node where the next layer's output will be written to
 *   @param in: The input node containing the input activations for the next layer
 *   @param shape_: The shape of the next layer's output
 *
 * @note
 *   - Returns nothing. The pointers are shifted accordingly.
 */
void shift(Node* out, Node* in, std::vector<size_t> shape_){

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
 * Shifts the pointer location for in and out activations/grad in order to process the previous layer in gpt (used for the backward pass)
 *
 * Args:
 *   @param out: The output node where the next layer's output will be written to
 *   @param in: The input node containing the input activations for the next layer
 *   @param shape_: The shape of the previous layer's input
 *
 * @note
 *   - Returns nothing. The pointers are shifted accordingly.
 */
void shift_back(Node* out, Node* in, std::vector<size_t> shape_){

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
 * @brief Encodes the input sequence and outputs the encoded representation.
 *
 * @param[out] out The output node containing the encoded representation.
 * @param[in] in The input node containing the sequence to be encoded.
 *
 * @note The input node's shape should be [B, T], where B is the batch size and T is the sequence length.
 * The output node's shape will be [B, T, C], where C is the embedding size.
 *
 * @note
 * This function applies the embedding lookup for both tokens and positions and then sums them up to get the final encoded representation.
 * The token embeddings are stored in the 'params' array, with the first C * vocab_size elements representing the token embeddings.
 * The position embeddings are stored in the rest of the 'params' array, with the next C * maxT elements representing the position embeddings.
 * The encoded representation is stored in the 'act' array of the output node.
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
 * @brief Performs the backward pass through the encoder layer.
 *
 * This function updates the gradient of token embeddings (wte) and positional embeddings (wpe) based on the gradients of the output (@p out).
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
 * @throw std::invalid_argument If @p C is not divisible by @p NH. This is a fundamental requirement of the Transformer architecture.
 * 
 * @note 
 * - The constructor dynamically allocates memory for various internal nodes and layers. Hence, it is crucial to handle exceptions during construction.
 */
TransformerBlock::TransformerBlock(float* params_, float* grad_, size_t C, size_t NH):
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
        layer_param += C * 3*C;
        layer_grad +=  C * 3*C;


        ra1 = new RowAdd(layer_param, layer_grad);
        layer_param += 3*C;
        layer_grad += 3*C;

        att = new Attention(NH);

        mat2 = new Matmul(layer_param, layer_grad);
        layer_param += C*C;
        layer_grad +=  C*C;

        ra2 = new RowAdd(layer_param, layer_grad);
        layer_param += C;
        layer_grad += C;

        res1 = new Add();

        ln2 = new LayerNorm(layer_param, layer_grad);
        layer_param += C + C;
        layer_grad += C + C;

        mat3 = new Matmul(layer_param, layer_grad);
        layer_param += C * 4*C;
        layer_grad += C * 4*C;

        ra3 = new RowAdd(layer_param, layer_grad);
        layer_param += 4*C;
        layer_grad += 4*C;

        gelu = new GELU();

        mat4 = new Matmul(layer_param, layer_grad);
        layer_param += 4*C * C;
        layer_grad += 4*C * C;

        ra4 = new RowAdd(layer_param, layer_grad);
        layer_param += C;
        layer_grad += C;

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

    Node* in_internal = new Node();
    in_internal->act = in->act;
    in_internal->act_grads = in->act_grads;
    in_internal->shape = in->shape;
    in_internal->size = in->size;
    in_internal->current_T = in->current_T;


    Node* out_internal = new Node();
    out_internal->act = in_internal->act + B * T * C;
    out_internal->act_grads = in_internal->act_grads + B * T * C;
    out_internal->shape = {B, T, C};
    out_internal->size = B * T * C;
    out_internal->current_T = in->current_T;


    // out should be of the same shape as in, should place a check here.

    // first layer norm
    ln1->forward(out_internal, in_internal);

    // first matmul and bias
    shift(out_internal, in_internal, {B, T, 3*C});
    mat1->forward(out_internal, in_internal);
    ra1->forward(out_internal, out_internal); // in place
    
    // attention
    shift(out_internal, in_internal, {B, T, C});
    att->forward(out_internal, in_internal);

    // second matmul and bias
    shift(out_internal, in_internal, {B, T, C});
    mat2->forward(out_internal, in_internal);
    ra2->forward(out_internal, out_internal); // in place

    // residual
    res1->forward(out_internal, res1_node); // in place

    // store activations for second residual
    res2_node->act = out_internal->act;
    res2_node->act_grads = out_internal->act_grads;
    res2_node->shape = out_internal->shape;
    res2_node->size = out_internal->size;

    // second layer norm
    shift(out_internal, in_internal, {B, T, C});
    ln2->forward(out_internal, in_internal);

    // third matmul and bias
    shift(out_internal, in_internal, {B, T, 4*C});
    mat3->forward(out_internal, in_internal); // (B, T, C) -> (B, T, 4*C)
    ra3->forward(out_internal, out_internal); // in place (B, T, 4*C) -> (B, T, 4*C)

    // GELU
    shift(out_internal, in_internal, {B, T, 4*C}); 
    gelu->forward(out_internal, in_internal); // (B, T, 4C) - > (B, T, 4C)

    // fourth matmul and bias
    shift(out_internal, in_internal, {B, T, C});
    mat4->forward(out_internal, in_internal);
    ra4->forward(out_internal, out_internal); // in place (B, T, 4C) -> (B, T, C)


    // second residual
    res2->forward(out_internal, res2_node); // in place (B, T, C) -> (B, T, C)


    // verify that results are in out Node
    if ((out_internal->act != out->act) || (out_internal->act_grads != out->act_grads) || (out_internal->size != out->size)){
        throw std::runtime_error("out node and out_internal node are not equal in transformer block");
    }

    // free memory of helper nodes
    delete in_internal;
    delete out_internal;

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

    Node* out_internal = new Node();
    out_internal->act = out->act;
    out_internal->act_grads = out->act_grads;
    out_internal->shape = out->shape;
    out_internal->size = out->size;

    Node* in_internal = new Node();
    in_internal->act = out->act - B * T * 4*C;
    in_internal->act_grads = out->act_grads - B * T * 4*C;
    in_internal->shape = {B, T, 4*C};
    in_internal->size = B * T * 4*C;

    // backward through second residual
    res2->backward(out_internal, res2_node);

    // backward through fourth matmul and bias
    ra4->backward(out_internal, out_internal);
    mat4->backward(out_internal, in_internal);

    // backward through GELU
    shift_back(out_internal, in_internal, {B, T, 4*C}); // (B, T, 4*C) is input shape of GELU operation
    gelu->backward(out_internal, in_internal);

    // backward through third matmul and bias
    shift_back(out_internal, in_internal, {B, T, C}); // (B, T, C) is input shape of matmul operation
    ra3->backward(out_internal, out_internal);
    mat3->backward(out_internal, in_internal);

    // backward through second layernorm
    shift_back(out_internal, in_internal, {B, T, C}); // (B, T, C) is the input shape of layernorm operation
    ln2->backward(out_internal, in_internal);

    // backward through first residual
    shift_back(out_internal, in_internal, {B, T, C}); // (B, T, C) is input shape of res1
    res1->backward(out_internal, res1_node);

    // backward through second matmul and bias
    ra2->backward(out_internal, out_internal);
    mat2->backward(out_internal, in_internal);

    // backward through attention
    shift_back(out_internal, in_internal, {B, T, 3*C}); // (B, T, 3*C) is input shape of attention
    att->backward(out_internal, in_internal);

    // backward through first matmul and bias
    shift_back(out_internal, in_internal, {B, T, C}); // (B, T, C) is input shape of matmul operatnion
    ra1->backward(out_internal, out_internal);
    mat1->backward(out_internal, in_internal);

    // backward through first layernorm
    shift_back(out_internal, in_internal, {B, T, C}); // (B, T, C) is input shape of layernorm
    ln1->backward(out_internal, in_internal);

    // verify that results are in 'in' Node
    if ((in_internal->act != in->act) || (in_internal->act_grads != in->act_grads) || (in_internal->size != in->size)){
        std::string errorMessage = "in node and in_internal node are not equal. Difference of pointer locations: " + std::to_string(in_internal->act - in->act);
        throw std::runtime_error(errorMessage);
        
    }

    // free allocated memory for helper nodes
    delete out_internal;
    delete in_internal;

}

/**
 * @brief Destructor for transformer block. Frees resources associated with block.
 *
 */
TransformerBlock::~TransformerBlock(){
    delete res2;
    delete ra4; delete mat4;
    delete gelu;
    delete ra3; delete mat3;
    delete ln2;
    delete res1;
    delete ra2; delete mat2;
    delete att;
    delete ra1; delete mat1;
    delete ln1;

    delete res1_node;
    delete res2_node;


}

/**
 * @brief Performs a forward pass through the attention block.
 *
 * @param[out] out  The output node.
 * @param[in]  in   The input node with shape `(B, T, C)`.
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
    // delete[] buffer should be called after yapping is done or after one training iteration is done
    
    
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
 * Forward pass of the GELU (Gaussian Error Linear Unit) activation function.
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
 * Backward pass of the GELU (Gaussian Error Linear Unit) activation function.
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
 * Forward pass of the Layernorm operation
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
 * Backward pass of the Layernorm operation
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
 * Computes the forward pass of matrix multiplication.
 *
 * This method performs a matrix multiplication operation on the input node's
 * activation values and the layer's parameters, storing the result in the output
 * node's activation values.
 *
 * @param out The output node where the result of the matrix multiplication will be stored.
 * @param in The input node containing the activation values to be multiplied.
 *
 * @throws std::invalid_argument If either the input or output node's activation values are nullptr.
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

    }
}




/**
 * Computes the forward pass of matrix multiplication assuming the weights are given tranposed.
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

    }
}

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

        // gradient wrt input
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    m, k, n, alpha, dLdC, n, B_, n, beta, dLdA, k);
        // gradient wrt parameters
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    p, n, m, alpha, A, k, dLdC, n, beta, dLdB, n);

    }
}

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

void Softmax::forward(Node* out, Node* in){
    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t V = in->shape[2];

    size_t current_T = in->current_T;

    for (size_t b = 0; b < B; b++){
        for (size_t t = 0; t < current_T; t++){
            float* bt_arr = in->act + b * T * V + t * V;
            float* out_bt_arr = out->act + b * T * V + t * V;

            // first pass to get maxval for numerical stability
            float maxval = -FLT_MAX;
            float val = 0.0f;
            for (size_t v = 0; v < V; v++){
                val = bt_arr[v];
                if (val > maxval){
                    maxval = val;
                }
            }
            // std::cout << maxval << std::endl;

            // second pass to compute exponentials
            float exp_sum = 0.0f;
            float exp_val = 0.0f;
            for (size_t v = 0; v < V; v++){
                exp_val = std::expf((bt_arr[v] - maxval)/temperature);
                out_bt_arr[v] = exp_val;
                exp_sum += exp_val;
            }

            float exp_sum_inv = exp_sum == 0.0f ? 0.0f : 1.0f / exp_sum;

            // third pass to normalize by sum
            for (size_t v = 0; v < V; v++){
                out_bt_arr[v] *= exp_sum_inv;
            }
        }
    }
}

void Softmax::backward(Node* out, Node* in){
    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t V = in->shape[2];

    float local_deriv;

    for (size_t b = 0; b < B; b++){
        for (size_t t = 0; t < T; t++){
            // seek to bt position 
            float* bt_out = out->act + b * T * V + t * V;
            float* bt_in_grad = in->act_grads + b * T * V + t * V;
            float* bt_out_grad = out->act_grads + b * T * V + t * V;

            for (size_t v = 0; v < V; v++){
                
                for (size_t i = 0; i < V ; i++){
                    local_deriv = bt_out[i] * ((i == v ? 1.0f : 0.0f) - bt_out[v]);
                    bt_in_grad[v] += bt_out_grad[i] * local_deriv / temperature;
                }
            }
        }
    }
}

void Softmax::set_temperature(float temp){
    temperature = temp;
}