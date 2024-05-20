#include <iostream>
#include <memory>
#include <gtest/gtest.h>
#include "src/classes.hpp"

using namespace std;

class AttentionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Common setup code here
    B = 1;
    T = 1;
    C = 768;
    maxT = 1024;


    in = new Node();
    in->act = new float[B * T * 3*C];
    in->act_grads = new float[B * T * 3*C];
    in->shape = {B, T, 3*C};
    in->size = B * T * 3*C;

    out = new Node();
    out->act = new float[B*T*C];
    out->act_grads = new float[B*T*C];
    out->shape = {B, T, C};
    out->size = B * T * C;
  }

  void TearDown() override {
    // Common teardown code here
    delete[] in->act;
    delete[] in->act_grads;
    delete in;

    delete[] out->act;
    delete[] out->act_grads;
    delete out;
  }

  size_t B;
  size_t T;
  size_t C;
  size_t maxT;
  Node* in;
  Node* out;
};

void fillArrayWithRandom(float* arr, int size) {
  for (int i = 0; i < size; i++) {
    arr[i] = (float)rand() / RAND_MAX; // Generates a random float between 0 and 1
  }
}

void fillArrayWithOnes(float* arr, int size) {
  for (int i = 0; i < size; i++) {
    arr[i] = 1.0; // Assigning each element the value 1
  }
}

TEST_F(AttentionTest, AttFlow) {
  size_t num_heads = 12;
  Attention* att = new Attention(num_heads, maxT);

  fillArrayWithRandom(in->act, in->size);

  att->forward(out, in);

  for (int i = 0; i < out->size; i++){
    EXPECT_GT(out->act[i], 0);
    EXPECT_LT(out->act[i], 1);
  }

  EXPECT_EQ(out->size, B*T*C);
  EXPECT_EQ(in->size, B*T*3*C);

  delete att;
}

TEST_F(AttentionTest, AllOnes) {
  size_t num_heads = 12;
  Attention* att = new Attention(num_heads, maxT);

  fillArrayWithOnes(in->act, in->size);

  att->forward(out, in);

  for (int i = 0; i < out->size; i++){
    EXPECT_EQ(out->act[i], 1.0);
  }

  delete att;
}

TEST_F(AttentionTest, InvalidShapes) {
  size_t num_heads = 12;
  Attention* att = new Attention(num_heads, maxT);

  in->size = B*T*C;
  EXPECT_THROW(att->forward(out, in), std::invalid_argument);

  delete att;
}



int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}