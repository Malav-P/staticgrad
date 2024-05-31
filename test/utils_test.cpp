#include <gtest/gtest.h>
#include "include/utils.hpp"

void fillArrayWithRandom(float* arr, int size) {
  for (int i = 0; i < size; i++) {
    arr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Generates a random float between -1 and 1
  }
}

void fillArrayWithOnes(float* arr, int size, float multiplier = 1.0f) {
  for (int i = 0; i < size; i++) {
    arr[i] = 1.0 * multiplier; // Assigning each element the value 1
  }
}

class UtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    B = 1;
    T = 1;
    V = 10;

    in = new Node();
    in->act = new float[B * T * V];
    in->act_grads = new float[B * T * V];
    in->shape = {B, T, V};
    in->size = B * T * V;

    out = new Node();
    out->act = new float[B * T * V];
    out->act_grads = new float[B * T * V];
    out->shape = {B, T, V};
    out->size = B * T * V;
  }

  void TearDown() override {
    delete[] in->act;
    delete[] in->act_grads;
    delete in;

    delete[] out->act;
    delete[] out->act_grads;
    delete out;
  }

  size_t B;
  size_t T;
  size_t V;
  Node* in;
  Node* out;
};


TEST_F(UtilsTest, crossentropyforward) {

    int targets[] = {8}; // token ID of the target. We only have one because B = T = 1 for the test
    float softmax_out[] = {0.12, 0.03, 0.25, 0.07, 0.08, 0.15, 0.04, 0.1, 0.09, 0.07};
    for (size_t i = 0 ; i < V; i++){
        in->act[i] = softmax_out[i];
    }

    EXPECT_NO_THROW(crossentropy_forward(out, in, targets));

    EXPECT_FLOAT_EQ(out->act[0], -std::logf(softmax_out[targets[0]]));
}

TEST_F(UtilsTest, crossentropy_softmax_backward) {

    int targets[] = {8}; // token ID of the target. We only have one because B = T = 1 for the test
    float softmax_out[] = {0.12, 0.03, 0.25, 0.07, 0.08, 0.15, 0.04, 0.1, 0.09, 0.07};
    for (size_t i = 0 ; i < V; i++){
        out->act[i] = softmax_out[i];
    }

    EXPECT_NO_THROW(crossentropy_softmax_backward(out, in, targets));

    for(int i = 0; i < V; i++){
        float indicator = (i == targets[0]) ? 1.0f : 0.0f;
        float expected = (softmax_out[i] - indicator) / (B*T);
        EXPECT_FLOAT_EQ(in->act_grads[i], expected);
    }
    
}



// Main function to run the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}