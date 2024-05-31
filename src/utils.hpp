#ifndef UTILS_HPP
#define UTILS_HPP

#include "./classes.hpp"

void crossentropy_forward(Node* out, Node* in, int* targets);

void crossentropy_softmax_backward(Node* out, Node* in, int* targets);


#endif // UTILS_HPP