#ifndef UTILS_HPP
#define UTILS_HPP

#include "node.hpp"

void crossentropy_forward(Node* out, Node* in, u_int16_t* targets);

void crossentropy_softmax_backward(Node* out, Node* in, u_int16_t* targets, float temperature);

u_int16_t sample_token(float* probabilities, size_t length, bool random = false);


#endif // UTILS_HPP