#ifndef UTILS_HPP
#define UTILS_HPP

#include "./classes.hpp"
#include <random>

void crossentropy_forward(Node* out, Node* in, int* targets);

void crossentropy_softmax_backward(Node* out, Node* in, int* targets, float temperature);

int sample_token(float* probabilities, int length, bool random = false);

#endif // UTILS_HPP