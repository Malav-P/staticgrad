#include <gtest/gtest.h>
#include "src/classes.hpp"

void fillArrayWithRandom(float* arr, int size) {
  for (int i = 0; i < size; i++) {
    arr[i] = (float)rand() / RAND_MAX; // Generates a random float between 0 and 1
  }
}

class SoftmaxTest : public ::testing::Test {
	protected:
		void SetUp() override {
			B = 2;
			T = 3;
			V = 4;

			in = new Node();
			in->act = new float[B * T * V];
			in->shape = {B, T, V};
			in->size = B * T * V;

			out = new Node();
			out->act = new float[B*T*V];
			out->shape = {B, T, V};
			out->size = B * T * V;
		}

		void TearDown() override {
			delete[] in->act;
			delete in;

			delete[] out->act;
			delete out;
		}

		size_t B;
		size_t T;
		size_t V;
		Node* in;
		Node* out;
};

TEST_F(SoftmaxTest, RandomInput) {
	Softmax* softmax = new Softmax();

	fillArrayWithRandom(in->act, in->size);

	softmax->forward(out, in);

	// Check that output values are probabilities (i.e., in range [0, 1])
	for (int i = 0; i < out->size; i++){
		EXPECT_GE(out->act[i], 0.0f);
		EXPECT_LE(out->act[i], 1.0f);
	}

	// Check that each row sums to 1
	for (size_t b = 0; b < B; b++) {
		for (size_t t = 0; t < T; t++) {
			float sum = 0.0f;
			for (size_t v = 0; v < V; v++) {
				sum += out->act[b * T * V + t * V + v];
			}
			EXPECT_NEAR(sum, 1.0f, 1e-6);
		}
	}

	delete softmax;
}

TEST_F(SoftmaxTest, LargeNegativeActivation) {
	Softmax* softmax = new Softmax();

	// Initialize input data such that the sum of all exponentials is zero
	for (int i = 0; i < in->size; i++){
		in->act[i] = -1000.0f;  // a large negative value will result in exp(x) being close to zero
	}

	softmax->forward(out, in);

	// Check that output values are all the same
	for (int i = 0; i < out->size; i++){
		EXPECT_NEAR(out->act[i], 1.0f / V, 1e-6);
	}

	delete softmax;
}

TEST_F(SoftmaxTest, BackwardPass) {
	out->act_grads = new float[out->size];
	in->act_grads = new float[in->size];

	Softmax* softmax = new Softmax();

	// Initialize out data 
	for (size_t i = 0; i < in->size; i++){
		out->act[i] = 0.25f; 
		out->act_grads[i] = 1.0f;
		in->act_grads[i] = 0.0f;
	}

	softmax->backward(out, in);

	// Check that output values are all the same
	float expected = 0.25f*(1.0f - 0.25f) + 0.25f*(-0.25f)*3.0f;

	for (size_t i = 0; i < in->size; i++){
		EXPECT_NEAR(in->act_grads[i], expected, 1e-6);
	}

	delete softmax;
	delete[] out->act_grads;
	delete[] in->act_grads;
}

TEST_F(SoftmaxTest, BackwardPassRandom) {

	in->act_grads= new float[B*T*V];
	out->act_grads = new float[B*T*V];

	Softmax* softmax = new Softmax();

	float arr[] = {0.2f, 0.3f, 0.4f, 0.1f};

	// Initialize out data 
	for (int i = 0; i < out->size; i++){
		out->act[i] = arr[i%4]; 
		out->act_grads[i] = 1.0f;
		in->act_grads[i] = 0.0f;
	}

	softmax->backward(out, in);

	// Check that output values are all the same
	float jac[] = {0.2f * (1.f - 0.2f), 0.3f * (-0.2f), 0.4f * (-0.2f), 0.1f * (-0.2f),
				   0.2f * (-0.3f), 0.3f * (1.f - 0.3f), 0.4f * (-0.3f), 0.1f * (-0.3f),
				   0.2f * (-0.4f), 0.3f * (-0.4f), 0.4f * (1.f - 0.4f), 0.1f * (-0.4f),
				   0.2f * (-0.1f), 0.3f * (-0.1f), 0.4f * (-0.1f), 0.1f * (1.f - 0.1f)};

	for (int i = 0; i < out->size; i++){
		float expected = 0.0f;
		for (int j = 0; j < out->size; j++){
			expected += jac[4*(i%4) + (j%4)];
		}
		EXPECT_NEAR(in->act_grads[i], expected, 1e-6);
	}

	delete softmax;
	delete[] out->act_grads;
	delete[] in->act_grads;
}

// Main function to run the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}