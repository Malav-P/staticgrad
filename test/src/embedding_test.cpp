#include <gtest/gtest.h>
#include "classes.hpp"
#include "test_common.hpp"

class EmbeddingTest : public ::testing::Test {
	protected:
		void SetUp() override {
			B = 2;
			T = 3;
			vocab_size = 8;
            C = 2;

			in = new Node();
			setup_node(in, {B, T});

			out = new Node();
			setup_node(out, {B, T, C});
		}

		void TearDown() override {
			teardown_node(out);
			teardown_node(in);
		}

		size_t B;
		size_t T;
        size_t C;

		size_t vocab_size;
		
		Node* in;
		Node* out;
};

TEST_F(EmbeddingTest, Forward) {


    size_t maxT = T + 1;

    float* params = new float[vocab_size*C + maxT*C];
    fillArrayWithRandom(params, vocab_size*C + maxT*C);

	Embedding* encoder = new Embedding(params, nullptr, C, vocab_size);
	float* p = (float*)encoder->params;


	// fill input with tokens
    for (size_t i = 0; i < in->size; i++){
        in->act[i] = i;
    }

	encoder->forward(out, in);

	for (size_t i = 0; i < in->size; i++){
        float* token_embed = p + i*C;
        float* pos_embed = p + C * encoder->vocab_size + i%T * C;
        float* out_bt = out->act + i*C;
        for (size_t j = 0; j < C; j++){
            EXPECT_FLOAT_EQ(token_embed[j] + pos_embed[j], out_bt[j]);
        }
	}

	delete encoder;
    delete[] params;
}

TEST_F(EmbeddingTest, Backward) {


    size_t maxT = T + 1;

    float* grad_ = new float[vocab_size*C + maxT*C]{0};

	Embedding* encoder = new Embedding(nullptr, grad_, C, vocab_size);
	float* p = (float*)encoder->params;
	float* g = (float*)encoder->grad;

    float* wte_g = g;
	float* wpe_g = g + C*vocab_size;

	// fill output grad with random
    fillArrayWithRandom(out->act_grads, out->size);

	// fill input with tokens
    for (size_t i = 0; i < in->size; i++){
        in->act[i] = i;
    }

	encoder->backward(out, in);

	for (size_t t = 0; t < T; t++){
		// seek to position emebedding, wpe_g is (maxT, C)
		float* pos_embed_g = wpe_g + t*C;
		for (size_t i = 0; i < C; i++){
			float deriv = 0;
			for (size_t b = 0; b < B; b++){

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