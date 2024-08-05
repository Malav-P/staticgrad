#ifndef UTILS_HPP
#define UTILS_HPP

#include "./classes.hpp"

void crossentropy_forward(Node* out, Node* in, u_int16_t* targets);

void crossentropy_softmax_backward(Node* out, Node* in, u_int16_t* targets, float temperature);

u_int16_t sample_token(float* probabilities, size_t length, bool random = false);
u_int16_t sample_token2(float* probabilities, size_t length, bool random);

void load_weights(float* dest, const std::string& fname, int expected_bytes = -1);


#endif // UTILS_HPP