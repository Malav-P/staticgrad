#include "./utils.hpp"

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

void crossentropy_softmax_backward(Node* out, Node* in, int* targets){
    // softmax takes (B, T, V) -> (B, T, V)
    // cross entropy takes (B, T, V) -> (B, T)

    // out is output of softmax , shape (B, T, V)
    // in in input to softmax, shape (B, T, V)
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

                float grad = prob - indicator;

                in->act_grads[b*T*V + t*V + v] += grad / (B*T); // B*T normalizer is present because final loss is mean over a (B, T) array of losses that result from the cross_entropy function
            }

        }
    }
}