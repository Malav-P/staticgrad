#ifndef UTILS_HPP
#define UTILS_HPP

#include "node.hpp"
#include <cstdint>

void crossentropy_forward(Node* out, Node* in, uint16_t* targets);

void crossentropy_softmax_backward(Node* out, Node* in, uint16_t* targets, float temperature);

uint16_t sample_token(float* probabilities, size_t length, bool random = false);


#endif // UTILS_HPP