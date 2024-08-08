#include "test_common.hpp"
#include "node.hpp"
#include <random>

void fillArrayWithRandom(float* arr, int size) {
  for (int i = 0; i < size; i++) {
    arr[i] = ((float)rand() / RAND_MAX)*2.0f - 1.0f; // Generates a random float between -1 and 1
  }
}

void fillArrayWithOnes(float* arr, int size) {
  for (int i = 0; i < size; i++) {
    arr[i] = 1.0; // Assigning each element the value 1
  }
}

void setup_node(Node* node, std::vector<size_t> shape_){
    size_t T = shape_[1];
    node->current_T = T;

    size_t numel = 1;

    for (size_t element : shape_){
      numel *= element;
    }

    // Set up the output node with shape (1, 3, 1)
    delete[] node->act;
    delete[] node->act_grads;
    node->act = new float[numel];
    node->act_grads = new float[numel];
    node->shape = shape_;
    node->size = numel;
}

void teardown_node(Node* node){
      delete[] node->act;
      delete[] node->act_grads;
      delete node;

      node = nullptr;
}
