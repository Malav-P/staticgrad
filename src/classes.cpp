#include "./classes.hpp"
#include <cmath>

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

void shift_back(Node* out, Node* in, std::vector<size_t> shape_){

    out->act = in->act;
    out->act_grads = in->act_grads;
    out->shape = in->shape;
    out->size = in->size;

    in->act += in->size;
    in->act_grads += in->size;

    in->shape = shape_;

    size_t numel = 1;

    for (size_t element : shape_) {
        numel *= element;
    }

    in->size = numel;

}

void Encoder::forward(Node* out, Node* in){
    size_t B = in->shape[0];
    size_t T = in->shape[1];

    float* wte = params;
    float* wpe = params + C * vocab_size;

    for (size_t b = 0; b < B; b++){
        for (size_t t = 0; t < T; t++){
            // get token_id (lrint casts the double to a long int)
            long int token_id = lrint(in->act[b*T + t]);
            
            // seek to token embedding, wte is (C, V)
            float* token_embed = wte + token_id;

            // seek to position emebedding, wpe is (maxT, C)
            float* pos_embed = wpe + t*C;

            // seek to output position
            float* output = out->act + b*T*C + t*C;

            for (size_t i = 0; i < C; i++){
                output[i] = token_embed[i*vocab_size] + pos_embed[i];
            }
        }
    }
}

void Encoder::backward(Node* out, Node* in){

    // output is (B, T, C)
    // wte is (C, V)
    // wpe is (maxT, C)

    size_t B = in->shape[0];
    size_t T = in->shape[1];

    float* wte_g = grad;
    float* wpe_g = grad + C * vocab_size;

    for (size_t b = 0; b < B; b++){
        for (size_t t = 0; t < T; t++){
            // get token_id (lrint casts the double to a long int)
            long int token_id = lrint(in->act[b*T + t]);
            
            // seek to token embedding grad, wte_g is (C, V)
            float* token_embed_g = wte_g + token_id;

            // seek to position emebedding, wpe_g is (maxT, C)
            float* pos_embed_g = wpe_g + t*C;

            // seek to output position
            float* output_g = out->act_grads + b*T*C + t*C;

            for (size_t i = 0; i < C; i++){
                float deriv = output_g[i];
                token_embed_g[i*vocab_size] += deriv;
                pos_embed_g[i] += deriv;
            }
        }
    }
    
}

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

    Node* out_internal = new Node();
    out_internal->act = in_internal->act + B * T * C;
    out_internal->act_grads = in_internal->act_grads + B * T * C;
    out_internal->shape = {B, T, C};
    out_internal->size = B * T * C;

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

    // second layer norm
    shift(out_internal, in_internal, {B, T, C});
    ln2->forward(out_internal, in_internal);

    // store activations for second residual
    res2_node->act = out_internal->act;
    res2_node->act_grads = out_internal->act_grads;
    res2_node->shape = out_internal->shape;
    res2_node->size = out_internal->size;

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
        throw std::runtime_error("out node and out_internal node are not equal");
    }

    // free memory of helper nodes
    delete in_internal;
    delete out_internal;

}

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
    ra1->backward(out_internal, in_internal);
    mat1->backward(out_internal, in_internal);

    // backward through first layernorm
    shift_back(out_internal, in_internal, {B, T, C}); // (B, T, C is input shape of layernorm
    ln1->backward(out_internal, in_internal);

    // verify that results are in 'in' Node
    if ((in_internal->act != in->act) || (in_internal->act_grads != in->act_grads) || (in_internal->size != in->size)){
        throw std::runtime_error("in node and in_internal node are not equal");
    }

    // free allocated memory for helper nodes
    delete out_internal;
    delete in_internal;

}

void Attention::forward(Node* out, Node* in){ // (B, T, 3C) -> (B, T, C)

    if (in->size != 3 * out->size){
        throw std::invalid_argument("input must be of shape (B, T, 3C) and output must be of shape (B, T, C)");
    }

    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t C = in->shape[2] / 3;

    size_t head_size = C / num_heads;

    float* buffer1 = buffer;
    float* buffer2 = buffer + 1024; // +1024 because maxseqlen=1024
    
    for (size_t b = 0 ; b < B; b++){

        for (size_t t = 0; t < T; t++){

            for (size_t nh = 0 ; nh < num_heads; nh++){

                // get query vector
                float* q = in->act + b * T * 3*C + t * 3*C + nh * head_size;

                // get raw scores
                float maxval = -9999999.0f;
                for (size_t t2 = 0; t2 <= t; t2++){
                    // get key vector for t2 token
                    float* k_t2 = in->act + b * T * 3*C + t2 * 3*C + C + nh * head_size;

                    // dot key vector with query vector to get score
                    float score = 0.0f;
                    for (size_t i = 0; i < head_size; i++){
                        score += q[i] * k_t2[i];
                    }

                    score = score / sqrtf(head_size);

                    if (score > maxval){
                        maxval = score;
                    }   

                    buffer1[t2] = score;       

                }

                // normalize scores according to softmax
                float exp_sum = 0;
                for (size_t t2 = 0; t2 <= t; t2++){
                    float exp_score = expf(buffer1[t2] - maxval);
                    buffer2[t2] = exp_score;
                    exp_sum += exp_score;
                }

                float exp_sum_inv = exp_sum == 0 ? 0.0f : 1 / exp_sum;

                for (size_t t2 = 0; t2 <= t; t2++){
                    buffer2[t2] *= exp_sum_inv;
                }


                // get output vector
                float* out_act = out->act + b * T * C + t * C + nh * head_size;
                // zero the output
                for (size_t i = 0; i < head_size; i++){
                    out_act[i] = 0.0f;
                }

                // accumulate weighted scores from tokens
                for (size_t t2 = 0; t2 <= t; t2++){
                    // find value vector for t2 token
                    float* v_t2 = in->act + b * T * 3*C + t2 * 3*C + C + C + nh * head_size;

                    for (size_t i=0; i < head_size; i++){
                        out_act[i] += buffer2[t2] * v_t2[i];
                    }
                }
    
            }

        }
    }

}

void Attention::backward(Node* out, Node* in) {
    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t C = in->shape[2] / 3;

    size_t head_size = C / num_heads;

    float* dbuffer1 = dbuffer;
    float* dbuffer2 = dbuffer + 1024; // +1024 because maxseqlen=1024

    float* buffer1 = buffer;
    float* buffer2 = buffer + 1024;


    for (size_t b = 0 ; b < B; b++){

        for (size_t t = 0; t < T; t++){

            for (size_t nh = 0 ; nh < num_heads; nh++){

                // get output grad
                float* out_g = out->act_grads + b * T * C + t * C + nh * head_size;

                // backward through accumulation 
                for (size_t t2 = 0; t2 <= t; t2++){
                    // find value vector and its grad for t2 token
                    float* dv_t2 = in->act_grads + b * T * 3*C + t2 * 3*C + C + C + nh * head_size;
                    float* v_t2  = in->act       + b * T * 3*C + t2 * 3*C + C + C + nh * head_size;

                    for (size_t i=0; i < head_size; i++){
                        dbuffer2[t2] += out_g[i] * v_t2[i];
                        dv_t2[t2] += out_g[i] * buffer2[t2];
                    }
                }

                // backward through softmax buffer2 = softmax(buffer1)
                for (size_t t2 = 0; t2 <= t; t2++){
                    for (size_t t3 = 0; t3 <= t; t3++){
                        dbuffer1[t2] += dbuffer2[t3] * buffer2[t3] * ((t2 == t3 ? 1.0f : 0.0f) - buffer2[t2] ); 
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
                        dq[i] += dbuffer1[t2] * k_t2[i] / sqrtf(head_size);
                        dk_t2[i] += dbuffer1[t2] * q[i] / sqrtf(head_size);
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

void LayerNorm::forward(Node* out, Node* in) { // (B, T, C) -> (B, T, C)


    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t C = in->shape[2];


    delete[] rstd; delete[] m;
    rstd = new float[B*T];
    m = new float[B*T];


    float eps = 1e-5; // division by zero prevention during normalization


    for (size_t b = 0; b < B; b++){
        for (size_t t = 0; t < T; t++){
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

void LayerNorm::backward(Node* out, Node* in){


    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t C = in->shape[2];


    float* w_g = grad;
    float* b_g = grad+C;

    float* w = params;
    float* b = params+C;

    for (size_t b = 0; b < B; b++){
        for (size_t t = 0; t < T; t++){
            float* out_g = out->act_grads + b*T*C + t*C;
            float* in_g = in->act_grads + b*T*C + t*C;
            float* inp = in->act + b*T*C + t*C;


            float m_ = m[b*T + t];
            float rstd_ = rstd[b*T + t];

            // first pass to store some useful values.
            float norm_g_mean = 0;
            float norm_g_mean_times_norm = 0;
            for (size_t i = 0; i < C; i++){
                float n = rstd_*(inp[i] - m_);
                float dnorm = out_g[i] * w[i];

                norm_g_mean += dnorm;

                norm_g_mean_times_norm += dnorm * n;
            }

            norm_g_mean = 1/C * norm_g_mean;
            norm_g_mean_times_norm = 1/C * norm_g_mean_times_norm;

            for (size_t i = 0; i < C; i++){
                // bias gradient
                b_g[i] += out_g[i];
                
                // scale gradient
                w_g[i] += (inp[i] - m_)*rstd_ * out_g[i];

                // input gradient
                float dnorm = out_g[i] * w[i];
                float n = rstd_*(inp[i] - m_);
                inp[0] += rstd_ * (dnorm - norm_g_mean - norm_g_mean_times_norm * n);
            }

        }
    }
}
        
void Matmul::forward(Node* out, Node* in) {

    if ((in->act == nullptr) || (out->act == nullptr)){
        throw std::invalid_argument("in or out activations are nullptrs.");
    }

    size_t B = in->shape[0];
    size_t T = in->shape[1],  C = in->shape[2];
    size_t p = C,  n = out->shape[2];
    
    float alpha = multiplier, beta = 0.0f;
    float* B_ = params;

    // for debugging, can remove
    // for (int i = 0; i < in->size; i++){
    //     std::cout<< in->act[i] << "\n";
    // }
    // std::cout<< B << "," << T << "," << C << "\n";


    for (size_t b = 0; b < B; b++){

        float* A = in->act + b * T * C;
        float* out_ = out->act + b * (T) * (n);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    T, n, C, alpha, A, C, B_, n, beta, out_, n);

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
        float* dLdC = out->act_grads + b * (m) * (n);


        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    m, k, n, alpha, dLdC, n, B_, n, beta, dLdA, k);
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    p, n, m, alpha, A, k, dLdC, n, beta, dLdB, n);

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

    size_t numel = in->size;

    for (size_t i = 0; i < numel ; i++){
        in->act_grads[i] += out->act_grads[i];
    }
}



void RowAdd::forward(Node* out, Node* in){

    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t C = in->shape[2];

    float* inp1 = params;

    for (size_t b = 0; b < B; b++){

        float* out_ = out->act + b * T * C;
        float* inp0 = in->act + b * T * C;

        for (size_t t = 0; t < T; t++){
            for (size_t c = 0; c < C; c++){
                out_[t*C + c] = inp0[t*C + c] + inp1[c];
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

    for (size_t b = 0; b < B; b++){
        for (size_t t = 0; t < T; t++){
            float* bt_arr = in->act + b * T * V + t * V;
            float* out_bt_arr = out->act + b * T * V + t * V;

            // first pass to get maxval for numerical stability
            float maxval = -99999.9f;
            float val = 0.0f;
            for (size_t v = 0; v < V; v++){
                val = bt_arr[v];
                if (val > maxval){
                    maxval = val;
                }
            }

            // second pass to compute exponentials
            float exp_sum = 0.0f;
            float exp_val = 0.0f;
            for (size_t v = 0; v < V; v++){
                exp_val = expf(bt_arr[v] - maxval);
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
                    bt_in_grad[v] += bt_out_grad[i] * local_deriv;
                }
            }
        }
    }
}

