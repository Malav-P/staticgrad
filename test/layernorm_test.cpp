#include <gtest/gtest.h>
#include "src/classes.hpp"

void fillArrayWithRandom(float* arr, int size) {
  for (int i = 0; i < size; i++) {
    arr[i] = (float)rand() / RAND_MAX; // Generates a random float between 0 and 1
  }
}

class LayerNormTest : public ::testing::Test {
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

    params = new float[C];
    grad = new float[C];



    layer_norm = new LayerNorm(params, grad);
  }

  void TearDown() override {
    delete[] in->act;
    delete[] in->act_grads;
    delete in;

    delete[] out->act;
    delete[] out->act_grads;
    delete out;

    delete[] params;
    delete[] grad;

    delete layer_norm;
  }

  size_t B;
  size_t T;
  size_t C;
  Node* in;
  Node* out;
  float* params;
  float* grad;
  LayerNorm* layer_norm;
};

TEST_F(LayerNormTest, ForwardPass) {
  fillArrayWithRandom(in->act, in->size);

  for (int i = 0; i<C; i++){
      layer_norm->params[i] = 1.0f;
      layer_norm->params[i+C] = 0.0f;
  }

  layer_norm->forward(out, in);


  float mean = 0.0f;

  // calc mean
  for (int i = 0; i<C; i++){
    mean += out->act[i];
  }
  mean = mean/C;

  // calculate var
  float v = 0.0f;
  for (size_t i = 0 ; i < C; i++){
      float dev = out->act[i] - mean;
      v += dev*dev;
  }
  v = v/C;

  // Check that the output values are within the expected range
  EXPECT_NEAR(mean, 0.0f, 1e-5);
  EXPECT_NEAR(v, 1.0f, 1e-3);
}

TEST_F(LayerNormTest, ScaleShift) {
  fillArrayWithRandom(in->act, in->size);

  for (int i = 0; i<C; i++){
      layer_norm->params[i] = 2.0f;
      layer_norm->params[i+C] = 1.0f;
  }

  layer_norm->forward(out, in);

  float mean = 0.0f;

  // calc mean
  for (int i = 0; i<C; i++){
    mean += out->act[i];
  }
  mean = mean/C;

  // calculate var
  float v = 0.0f;
  for (size_t i = 0 ; i < C; i++){
      float dev = out->act[i] - mean;
      v += dev*dev;
  }
  v = v/C;

  // Check that the output values are within the expected range
  EXPECT_NEAR(mean, 1.0f, 1e-5);
  EXPECT_NEAR(v, 4.0f, 1e-3);

  // Check that the output values are scaled and shifted correctly
  for (int i = 0; i < C; i++) {
    float expected = layer_norm->params[i] * (layer_norm->rstd[0])*(in->act[i] - layer_norm->m[0]) + layer_norm->params[i + C];
    EXPECT_NEAR(out->act[i], expected, 1e-5);
  }
}

// TEST_F(LayerNormTest, NullData) {
//   delete[] in->act;
//   in->act = nullptr;

//   // This should throw an exception or return an error
//   EXPECT_THROW(layer_norm->forward(out, in), std::invalid_argument);
// }

// TEST_F(LayerNormTest, DifferentSize) {
//   delete[] in->act;
//   in->act = new float[100];
//   in->size = 100;
//   in->shape = {1, 100};

//   // This should throw an exception or return an error
//   EXPECT_THROW(layer_norm->forward(out, in), std::invalid_argument);
// }

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}