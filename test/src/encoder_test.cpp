#include <gtest/gtest.h>
#include "classes.hpp"
#include "test_common.hpp"

class EncoderTest : public ::testing::Test {
	protected:
		void SetUp() override {
			B = 2;
			T = 3;
			vocab_size = 8;
            C = 2;

			in = new Node();
			in->act = new float[B * T];
			in->act_grads = new float[B * T];
			in->shape = {B, T};
			in->size = B * T;

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
		size_t vocab_size;
        size_t C;
		Node* in;
		Node* out;
};

TEST_F(EncoderTest, Forward) {

    size_t maxT = T + 1;

    float* params = new float[vocab_size*C + maxT*C];
    fillArrayWithRandom(params, vocab_size*C + maxT*C);

	Encoder* encoder = new Encoder(params, nullptr, C, vocab_size);

	// fill input with tokens
    for (int i = 0; i < in->size; i++){
        in->act[i] = i;
    }

	encoder->forward(out, in);

	for (int i = 0; i < in->size; i++){
        float* token_embed = encoder->params + i*C;
        float* pos_embed = encoder->params + C * encoder->vocab_size + i%T * C;
        float* out_bt = out->act + i*C;
        for (int j = 0; j < C; j++){
            EXPECT_FLOAT_EQ(token_embed[j] + pos_embed[j], out_bt[j]);
        }
	}

	delete encoder;
    delete[] params;
}

TEST_F(EncoderTest, Backward) {

    size_t maxT = T + 1;

    float* grad_ = new float[vocab_size*C + maxT*C]{0};

	Encoder* encoder = new Encoder(nullptr, grad_, C, vocab_size);

    float* wte_g = encoder->grad;
	float* wpe_g = encoder->grad + C*vocab_size;

	// fill output grad with random
    fillArrayWithRandom(out->act_grads, out->size);

	// fill input with tokens
    for (int i = 0; i < in->size; i++){
        in->act[i] = i;
    }

	encoder->backward(out, in);

	for (int t = 0; t < T; t++){
		// seek to position emebedding, wpe_g is (maxT, C)
		float* pos_embed_g = wpe_g + t*C;
		for (size_t i = 0; i < C; i++){
			float deriv = 0;
			for (int b = 0; b < B; b++){

				// seek to output position
				float* output_g = out->act_grads + b*T*C + t*C;
				deriv += output_g[i];

				int tokenID = lrint(in->act[b*T + t]);
				float* token_embed_g = wte_g + tokenID*C;
            	EXPECT_FLOAT_EQ(token_embed_g[i], output_g[i]);

			}
			EXPECT_FLOAT_EQ(pos_embed_g[i], deriv);
		}
    
	}
	delete encoder;
    delete[] grad_;
}

// Main function to run the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}