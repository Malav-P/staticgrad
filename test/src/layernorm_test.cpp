#include <gtest/gtest.h>
#include "classes.hpp"
#include "test_common.hpp"
#include <cstring>


class LayerNormTest : public ::testing::Test {
 protected:
  void SetUp() override {
    B = 1;
    T = 1;
    C = 768;

    in = new Node();
    setup_node(in, {B, T, C});

    out = new Node();
    setup_node(out, {B, T, C});

    params = new float[C + C];
    grad = new float[C + C];
    
    layer_norm = new LayerNorm(params, grad);
  }

  void TearDown() override {
    teardown_node(out);
    teardown_node(in);

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

  float* p = (float*)layer_norm->params;

	for (size_t i = 0; i<C; i++){
		p[i] = 1.0f;
		p[i+C] = 0.0f;
	}

	layer_norm->forward(out, in);


	float mean = 0.0f;

	// calc mean
	for (size_t i = 0; i<C; i++){
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

  float* p = (float*)layer_norm->params;

  for (size_t i = 0; i<C; i++){
      p[i] = 2.0f;
      p[i+C] = 1.0f;
  }

  layer_norm->forward(out, in);

  float mean = 0.0f;

  // calc mean
  for (size_t i = 0; i<C; i++){
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
  for (size_t i = 0; i < C; i++) {
    float expected = p[i] * (layer_norm->rstd[0])*(in->act[i] - layer_norm->means[0]) + p[i + C];
    EXPECT_NEAR(out->act[i], expected, 1e-5);
  }
}


TEST_F(LayerNormTest, Backward){

  float* p = (float*)layer_norm->params;
  float* g = (float*)layer_norm->grad;

	// do forward pass

	fillArrayWithRandom(in->act, in->size);
	fillArrayWithRandom(p, 2*C);

	layer_norm->forward(out, in);

	// prepare for backward pass

	fillArrayWithOnes(out->act_grads, out->size);

	std::memset(g, 0, 2*C*sizeof(float));
	std::memset(in->act_grads, 0, in->size * sizeof(float));

	layer_norm->backward(out, in);

	// check shift (bias) gradients
	for (size_t i = 0; i < C; i++){
		float expected = 0.0f;
		for (size_t b = 0; b < B; b++){
			for (size_t t = 0; t < T; t++){
				expected += out->act_grads[b*T*C + t*C + i];
			}
		}
		EXPECT_FLOAT_EQ(g[C + i], expected);
	}

	// Check scale (weight) gradients
	for (size_t i = 0; i < C; i++){
		float expected = 0.0f;
		for (size_t b = 0; b < B; b++){
			for (size_t t = 0; t < T; t++){
				expected += out->act_grads[b*T*C + t*C + i] * (in->act[b*T*C + t*C + i] - layer_norm->means[b*T + t]) * layer_norm->rstd[b*T + t];
			}
		}
		EXPECT_FLOAT_EQ(g[i], expected);
	}

	// TODO Check input gradients

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}