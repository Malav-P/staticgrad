#include "utils.hpp"
#include <random>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <algorithm>

/**
 * Samples a token from a given probability distribution.
 *
 * 
 *   @param logits: A pointer to the logits, an array of length `length`.
 *   @param length: The number of elements in the probability distribution.
 *   @param random: A boolean flag indicating whether to sample randomly or deterministically.
 *
 *   @return `token`: The sampled token.
 *
 *   @throws `std::invalid_argument` if the probabilities are not valid (i.e., non-negative and summing up to 1).
 */
uint16_t sample_token(const float* logits, const size_t length, const bool random){

    float* probabilities = new float[length];

    float maxval = -FLT_MAX;
    for (size_t t2 = 0; t2 < length; t2++){
        
        float score = logits[t2];

        if (score > maxval){
            maxval = score;
        }   
    }

    float exp_sum = 0.0f;
    for (size_t t2 = 0; t2 < length; t2++){
        float exp_score = expf(logits[t2] - maxval);
        probabilities[t2] = exp_score;
        exp_sum += exp_score;
    }

    float exp_sum_inv = exp_sum == 0 ? 0.0f : 1.0f / exp_sum;

    for (size_t t2 = 0; t2 < length; t2++){
        probabilities[t2] *= exp_sum_inv;
    }

    // Check if probabilities are valid
    float sum = 0.0f;
    for (size_t i = 0; i < length; i++) {
        if (probabilities[i] < 0.0) {
            throw std::invalid_argument("Probabilities must be non-negative");
        }
        sum += probabilities[i];
    }

    if (std::abs(sum - 1.0f) > 1e-3) {
        throw std::invalid_argument("Probabilities must sum up to 1");
    }

    uint16_t token;

    if (random){
        static std::random_device rd;
        static std::mt19937 gen(rd());

        std::discrete_distribution<> d(probabilities, probabilities + length);
        token = d(gen);
    }

    else {
        token = std::distance(probabilities, std::max_element(probabilities, probabilities + length));
    }

    delete[] probabilities;

    return token;
    
}


/**
 * Computes the mean over all axes of loss tensor.
 *
 * 
 *   @param loss_node: The node where the loss is stored, a 2D tensor of shape (B, T).
 *
 * 
 *   @return `m_loss`: The average loss
 */
float mean_loss(Node* loss_node){
    float m_loss = 0.0f;
    size_t numel = loss_node->size;

    for (size_t i = 0; i < numel; i++){
        m_loss += loss_node->act[i];
        loss_node->act_grads[i] = 1.0f / numel;
    }

    m_loss /= numel;

    return m_loss;
}


/**
 * 
 * @brief Computes the softmax + cross-entropy loss for a given tensor of logits and targets.
 *
 *   @param out: The output node where the computed loss will be stored, a 2D tensor of shape (B, T).
 *   @param in: The input node containing the logits, a 3D tensor of shape (B, T, V).
 *   @param targets: An array of shape (B, T) containing the true tokenID for each sample.
 *
 */
void SoftmaxCrossEntropy::forward(Node* out, Node* in, const uint16_t* targets){
    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t V = in->shape[2];
    
    delete[] expsums; expsums = nullptr;
    delete[] maxvals; maxvals = nullptr;

    expsums = new float[B*T];
    maxvals = new float[B*T];

    for (size_t b = 0; b < B; b++){
        for (size_t t = 0; t < T; t++){
            float* logits = in->act + b*T*V + t*V;
            uint16_t target = targets[b*T + t];

            // compute z_true
            float target_logit = logits[target];

            // Find the maximum value in the array
            float max_val = -FLT_MAX;
            for (size_t i = 0; i < V; i++) {
                if (logits[i] > max_val) {
                    max_val = logits[i];
                }
            }
            // Compute the sum of exp(x - max_val)
            float sum_exp = 0.0;
            for (size_t i = 0; i < V; i++) {
                sum_exp += expf(logits[i] - max_val);
            }

            maxvals[b*T + t] = max_val;
            expsums[b*T + t] = sum_exp;

            // compute log sum exp
            float logsumexp = max_val + logf(sum_exp);

            out->act[b*T + t] = -target_logit + logsumexp;

        }
    }
}

/**
 * 
 * @brief Backpropagates the gradients from loss through the cross entropy and softmax functions. 
 *
 *   @param out: The output node where the computed loss was stored, a 2D tensor of shape (B, T).
 *   @param in: The input node containing the logits, a 3D tensor of shape (B, T, V).
 *   @param targets: An array of shape (B, T) containing the true tokenID for each sample.
 * 
 */
void SoftmaxCrossEntropy::backward(Node* out, Node* in, const uint16_t* targets){
    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t V = in->shape[2];
    
    float temperature = 1.0f;

    for (size_t b = 0; b < B; b++){
        for (size_t t = 0; t < T; t++){
            float* logits = in->act + b*T*V + t*V;
            float maxval = maxvals[b*T + t];
            float expsum = expsums[b*T + t];
            uint16_t target = targets[b*T + t];

            for (size_t v = 0; v < V; v++){
                float indicator = (v == target) ? 1.0f : 0.0f;
                float prob = expf(logits[v] - maxval) / expsum;

                float grad = (prob - indicator) / temperature;

                in->act_grads[b*T*V + t*V + v] += grad * out->act_grads[b*T + t]; // B*T normalizer is present because final loss is mean over a (B, T) array of losses that result from the cross_entropy function
            }

        }
    }

    delete[] maxvals; maxvals = nullptr;
    delete[] expsums; expsums = nullptr;
}
