#include <gtest/gtest.h>
#include "classes.hpp"
#include "test_common.hpp"

using namespace std;


class TransformerBlockTest : public ::testing::Test {
	protected:
	void SetUp() override {
		// Common setup code here
		B = 2;
		T = 3;
		C = 768;
		maxT = 1024;

		activations = new float[17*B*T*C];
		activations_grad = new float[17*B*T*C];

		params = new float[12*C*C + 13*C];
		params_grad = new float[12*C*C + 13*C];

		fillArrayWithRandom(params, 12*C*C + 13*C);

		in = new Node();
		in->act = activations;
		in->act_grads = activations_grad;
		in->shape = {B, T, C};
		in->size = B * T * C;

		out = new Node();
		out->act = activations + 16*B*T*C;
		out->act_grads = activations_grad + 16*B*T*C;
		out->shape = {B, T, C};
		out->size = B * T * C;
	}

	void TearDown() override {
		delete[] activations;
		delete[] activations_grad;
		delete[] params;
		delete[] params_grad;
		delete in;
		delete out;
	}

	size_t B;
	size_t T;
	size_t C;
	size_t maxT;
	Node* in;
	Node* out;
	float* activations;
	float* activations_grad;
	float* params;
	float* params_grad;
};

TEST_F(TransformerBlockTest, InvalidNumHeads) {
	size_t num_heads = 11; // 768 is not divisible by 11
	TransformerBlock* tblock = nullptr;
	EXPECT_THROW(tblock = new TransformerBlock(params, params_grad, C, num_heads), std::invalid_argument);

	num_heads = 0; // 768 is not divisible by 0
	EXPECT_THROW(tblock = new TransformerBlock(params, params_grad, C, num_heads), std::invalid_argument);
}


TEST_F(TransformerBlockTest, WarnSmallVariance) {
	size_t num_heads = 12;
	TransformerBlock* tblock = new TransformerBlock(params, params_grad, C, num_heads);

	std::memset(in->act, 0, in->size * sizeof(float));

	testing::internal::CaptureStderr();

	tblock->forward(out, in);

	std::string output = testing::internal::GetCapturedStderr();
	
	EXPECT_TRUE(output.find("Variance in layernorm close to zero.") != std::string::npos);

	delete tblock;
}


TEST_F(TransformerBlockTest, ForwardPass) {
	size_t num_heads = 12;
	TransformerBlock* tblock = new TransformerBlock(params, params_grad, C, num_heads);

	fillArrayWithRandom(in->act, 17*B*T*C);

	EXPECT_NO_THROW(tblock->forward(out, in));

    // Verify that the output shape is correct
    EXPECT_EQ(out->shape[0], in->shape[0]);
    EXPECT_EQ(out->shape[1], in->shape[1]);
    EXPECT_EQ(out->shape[2], in->shape[2]);
	

	delete tblock;
}

TEST_F(TransformerBlockTest, DoubleForward) {
	size_t num_heads = 12;
	TransformerBlock* tblock = new TransformerBlock(params, params_grad, C, num_heads);
	fillArrayWithRandom(in->act, in->size);

	tblock->forward(out, in);

	float* buffer = new float[out->size];
	for (size_t i = 0 ; i < out->size; i++){
		buffer[i] = out->act[i];
	}

	tblock->forward(out, in);

	for (size_t i = 0 ; i < out->size; i++){
		EXPECT_FLOAT_EQ(buffer[i], out->act[i]);
	}
	

	delete tblock;
	delete[] buffer;
}

TEST_F(TransformerBlockTest, Backward) {
	size_t num_heads = 12;
	TransformerBlock* tblock = new TransformerBlock(params, params_grad, C, num_heads);

	fillArrayWithRandom(in->act, in->size);
	tblock->forward(out, in);

	for (size_t j = 0; j < out->size ; j++){
		out->act_grads[j] = 1.0f / (out->size);
	}

	EXPECT_NO_THROW(tblock->backward(out, in));


	delete tblock;
}

TEST_F(TransformerBlockTest, ParameterAllocation) {
	size_t num_heads = 12;
	TransformerBlock* tblock = new TransformerBlock(params, params_grad, C, num_heads);

	int c = C;

	// first layernorm
	EXPECT_EQ(tblock->mat1->params - tblock->ln1->params, 2*c);
	EXPECT_EQ(tblock->mat1->grad - tblock->ln1->grad, 2*c);

	// first matmul
	EXPECT_EQ(tblock->ra1->params - tblock->mat1->params, 3*c*c);
	EXPECT_EQ(tblock->ra1->grad - tblock->mat1->grad, 3*c*c);

	// first row add
	EXPECT_EQ(tblock->mat2->params - tblock->ra1->params, 3*c);
	EXPECT_EQ(tblock->mat2->grad - tblock->ra1->grad, 3*c);

	// second matmul
	EXPECT_EQ(tblock->ra2->params - tblock->mat2->params, c*c);
	EXPECT_EQ(tblock->ra2->grad - tblock->mat2->grad, c*c);

	// second rowadd
	EXPECT_EQ(tblock->ln2->params - tblock->ra2->params, c);
	EXPECT_EQ(tblock->ln2->grad - tblock->ra2->grad, c);

	// second layernorm
	EXPECT_EQ(tblock->mat3->params - tblock->ln2->params, 2*c);
	EXPECT_EQ(tblock->mat3->grad - tblock->ln2->grad, 2*c);

	// third matmul
	EXPECT_EQ(tblock->ra3->params - tblock->mat3->params, 4*c*c);
	EXPECT_EQ(tblock->ra3->grad - tblock->mat3->grad, 4*c*c);

	// third rowadd
	EXPECT_EQ(tblock->mat4->params - tblock->ra3->params, 4*c);
	EXPECT_EQ(tblock->mat4->grad - tblock->ra3->grad, 4*c);

	// fourth matmul
	EXPECT_EQ(tblock->ra4->params - tblock->mat4->params, 4*c*c);
	EXPECT_EQ(tblock->ra4->grad - tblock->mat4->grad, 4*c*c);

	// fourth row add
	EXPECT_EQ(params + 12*C*C + 13*C - tblock->ra4->params, c);
	EXPECT_EQ(params_grad + 12*C*C + 13*C- tblock->ra4->grad, c);


	delete tblock;
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}