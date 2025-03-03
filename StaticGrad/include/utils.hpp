#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdint>
#include "node.hpp"

uint16_t sample_token(const float* logits, const size_t length, const bool random = false);

float mean_loss(Node* loss_node);

class SoftmaxCrossEntropy {
    public:
        float* expsums;
        float* maxvals;

        ~SoftmaxCrossEntropy(){
            delete[] expsums; delete[] maxvals;
        }

        void forward(Node* out, Node* in, const uint16_t* targets);
        void backward(Node* out, Node* in, const uint16_t* targets);
};


#endif // UTILS_HPP