#include <gtest/gtest.h>
#include "src/classes.hpp"

class AddTest : public ::testing::Test {
 protected:
  void SetUp() override {
    B = 1;
    T = 1;
    C = 768;

    in = new Node();
    in->act = new float[B * T * C];
    in->act_grads = new float[B * T * C];
    in->shape = {B, T, C};
    in->size = B * T * C;

    out = new Node();
    out->act = new float[B*T*C];
    out->act_grads = new float[B*T*C];
    out->shape = {B, T, C};
    out->size = B * T * C;
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
  size_t C;
  Node* in;
  Node* out;
};

void fillArrayWithRandom(float* arr, int size) {
  for (int i = 0; i < size; i++) {
    arr[i] = (float)rand() / RAND_MAX; // Generates a random float between 0 and 1
  }
}

void fillArrayWithOnes(float* arr, int size, float multiplier = 1.0f) {
  for (int i = 0; i < size; i++) {
    arr[i] = 1.0 * multiplier; // Assigning each element the value 1
  }
}

TEST_F(AddTest, ToArrayofOne) {
  Add* add = new Add();

  fillArrayWithRandom(in->act, in->size);
  fillArrayWithOnes(out->act, out->size);

  add->forward(out, in);

  for (int i = 0; i < out->size; i++){
    EXPECT_EQ(out->act[i], 1.0f + in->act[i]);
  }

  delete add;
}

TEST_F(AddTest, ToArrayofRandom) {
  Add* add = new Add();

  fillArrayWithRandom(in->act, in->size);
  fillArrayWithRandom(out->act, out->size);

  float* expected = new float[out->size];
  for (int i = 0; i < out->size; i++) {
    expected[i] = in->act[i] + out->act[i];
  }

  add->forward(out, in);

  for (int i = 0; i < out->size; i++){
    EXPECT_EQ(out->act[i], expected[i]);
  }

  delete[] expected;
  delete add;
}

TEST_F(AddTest, ToArrayofZeros) {
  Add* add = new Add();

  fillArrayWithRandom(in->act, in->size);
  fillArrayWithOnes(out->act, out->size, 0.0f); // Fill with zeros instead of ones

  add->forward(out, in);

  for (int i = 0; i < out->size; i++){
    EXPECT_EQ(out->act[i], in->act[i]);
  }

  delete add;
}

TEST_F(AddTest, NullData) {
  Add* add = new Add();

  delete[] in->act;
  in->act = nullptr;
  fillArrayWithRandom(out->act, out->size);

  // This should throw an exception or return an error
  EXPECT_THROW(add->forward(out, in), std::invalid_argument);

  delete add;
}

TEST_F(AddTest, DifferentSize) {
  Add* add = new Add();

  delete[] in->act;
  in->act = new float[100];
  in->size = 100;
  in->shape = {1, 100};
  fillArrayWithRandom(out->act, out->size);

  // This should throw an exception or return an error
  EXPECT_THROW(add->forward(out, in), std::invalid_argument);

  delete add;
}



// Main function to run the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}