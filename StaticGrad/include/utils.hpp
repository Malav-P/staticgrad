#ifndef UTILS_HPP
#define UTILS_HPP

#include "node.hpp"
#include <cstdint>

void crossentropy_forward(Node* out, Node* in, const uint16_t* targets);

void crossentropy_softmax_backward(Node* out, Node* in, const uint16_t* targets, const float temperature);

uint16_t sample_token(const float* probabilities, const size_t length, const bool random = false);


#endif // UTILS_HPP