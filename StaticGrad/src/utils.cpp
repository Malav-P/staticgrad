#include "../include/utils.hpp"

void crossentropy_forward(Node* out, Node* in, int* targets){ // (B, T, V) -> (B, T)

    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t V = in->shape[2];


    for (size_t b = 0; b < B; b++){
        for (size_t t = 0; t < T; t++){
            float* probabilities = in->act + b*T*V + t*V;
            int target = targets[b*T + t];

            out->act[b*T + t] = -std::logf(probabilities[target]);

        }
    }
}

/**
 * Computes the gradients of the cross-entropy loss with respect to the input
 * to the softmax function.
 *
 * Args:
 *   out: The output of the softmax function, a 3D tensor of shape (B, T, V).
 *   in: The input to the softmax function, a 3D tensor of shape (B, T, V).
 *   targets: An array of shape (B, T) containing the true tokenID for each sample.
 *   temperature: the temperature parameter
 *
 * Returns:
 *   None. The gradients are accumulated in the `act_grads` array of the `in` node.
 */
void crossentropy_softmax_backward(Node* out, Node* in, int* targets, float temperature){
    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t V = in->shape[2];

    for (size_t b = 0; b < B; b++){
        for (size_t t = 0; t < T; t++){
            float* probabilities = out->act + b*T*V + t*V;
            int target = targets[b*T + t];

            for (size_t v = 0; v < V; v++){
                float indicator = (v == target) ? 1.0f : 0.0f;
                float prob = probabilities[v];

                float grad = (prob - indicator) / temperature;

                in->act_grads[b*T*V + t*V + v] += grad / (B*T); // B*T normalizer is present because final loss is mean over a (B, T) array of losses that result from the cross_entropy function
            }

        }
    }
}

int sample_token(float* probabilities, int length, bool random){

    // Check if probabilities are valid
    float sum = 0.0;
    for (int i = 0; i < length; i++) {
        if (probabilities[i] < 0.0) {
            throw std::invalid_argument("Probabilities must be non-negative");
        }
        sum += probabilities[i];
    }

    if (std::abs(sum - 1.0) > 1e-6) {
        throw std::invalid_argument("Probabilities must sum up to 1");
    }

    if (random){
        static std::random_device rd;
        static std::mt19937 gen(rd());

        std::discrete_distribution<> d(probabilities, probabilities + length);
        return d(gen);
    }

    else {
        return std::distance(probabilities, std::max_element(probabilities, probabilities + length));

    }


    
}