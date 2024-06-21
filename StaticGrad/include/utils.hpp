#ifndef UTILS_HPP
#define UTILS_HPP

#include "./classes.hpp"
#include <random>
#include <fstream>

void crossentropy_forward(Node* out, Node* in, u_int16_t* targets);

void crossentropy_softmax_backward(Node* out, Node* in, u_int16_t* targets, float temperature);

int sample_token(float* probabilities, int length, bool random = false);

void load_weights(float* dest, const std::string& fname, int expected_bytes = -1);


#endif // UTILS_HPP