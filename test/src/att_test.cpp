#include <gtest/gtest.h>
#include "classes.hpp"
#include "test_common.hpp"

using namespace std;


class AttentionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Common setup code here
    B = 2;
    T = 3;
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

TEST_F(AttentionTest, Forward) {
	size_t num_heads = 12;
	Attention* att = new Attention(num_heads, maxT);

	int half_buffer_size = num_heads*B*lrint(T*(T+1)/2);
	int head_size = C / att->num_heads;


	// fill query, key, value
	for (int b = 0; b < B; b++){
		for (int t = 0; t < T; t++){
			for (int i = 0; i < C; i++){
				// fill query
				in->act[b*T*3*C + t*3*C + i] = 1.0f / C * (i+1);

				// fill key
				in->act[b*T*3*C + t*3*C + C + i] = 2.0f / C * (i+1);

				// fill value
				in->act[b*T*3*C + t*3*C + C + C + i] = 3.0f / C;
			}
		}
	}

	att->forward(out, in);

	float* tracker = att->buffer + half_buffer_size;
	float* start = tracker;

	for (int b = 0; b < B; b++){
		for (int t = 0; t < T; t++){
			for (int nh = 0; nh < att->num_heads; nh++){

				float* buffer2 = att->buffer + half_buffer_size + b*num_heads*lrint(T*(T+1)/2) + lrint(t*(t+1)/2)*num_heads + (t + 1)*nh;
				tracker += (t + 1);

				float sum = 0.0f;
				for (int t2 = 0; t2 <= t; t2++){
					sum += buffer2[t2];
				}

				EXPECT_NEAR(sum, 1.0f, 1e-5);

                // get output vector
                float* out_act = out->act + b * T * C + t * C + nh * head_size;

				float* expected = new float[head_size];
                for (size_t t2 = 0; t2 <= t; t2++){
                    // find value vector for t2 token
                    float* v_t2 = in->act + b * T * 3*C + t2 * 3*C + C + C + nh * head_size;

                    for (size_t i=0; i < head_size; i++){
                        expected[i] += buffer2[t2] * v_t2[i];
					}

                }

				for (size_t i=0; i < head_size; i++){
                    EXPECT_FLOAT_EQ(expected[i], out_act[i]);
                }

				delete[] expected;

			}
		}
	}

	EXPECT_EQ(tracker - start, half_buffer_size); // ensure we've traversed all of the allocated buffer

	EXPECT_EQ(out->size, B*T*C);
	EXPECT_EQ(in->size, B*T*3*C);

	delete att;
}

// TODO Test backward pass

TEST_F(AttentionTest, Backward){

	// ----- to fill buffers

	size_t num_heads = 12;
	Attention* att = new Attention(num_heads, maxT);

	int half_buffer_size = num_heads*B*lrint(T*(T+1)/2);
	int head_size = C / att->num_heads;


	// //fill query, key, value
	// for (int b = 0; b < B; b++){
	// 	for (int t = 0; t < T; t++){
	// 		for (int i = 0; i < C; i++){
	// 			// fill query
	// 			in->act[b*T*3*C + t*3*C + i] = 1.0f / C * (i+1);

	// 			// fill key
	// 			in->act[b*T*3*C + t*3*C + C + i] = 2.0f / C * (i+1);

	// 			// fill value
	// 			in->act[b*T*3*C + t*3*C + C + C + i] = 3.0f / C;
	// 		}
	// 	}
	// }
	fillArrayWithRandom(in->act, in->size);

	// calling att->backward before att->forward should throw
	EXPECT_THROW(att->backward(out, in), std::runtime_error);

	att->forward(out, in);


	// ------- Begin testing backwards ---- 

	fillArrayWithOnes(out->act_grads, out->size);
	std::memset(in->act_grads, 0, in->size * sizeof(float));

	for (int i = 0 ; i < in->size; i++){
		EXPECT_FLOAT_EQ(in->act_grads[i], 0.0f);
	}

	att->backward(out, in);

	for (size_t b = 0; b < B; b++){
		for (size_t t = 0; t < T; t++){
			for (size_t nh = 0; nh < num_heads; nh++){
				float* dq = in->act_grads + b * T * 3*C + t * 3*C + nh * head_size;
				
				if (t == 0){
					for (int i = 0; i < head_size; i++){
						EXPECT_FLOAT_EQ(dq[i], 0.0f);
					}

				}
				else{
					for (int i = 0; i < head_size; i++){
						EXPECT_NE(dq[i], 0.0f);
					}
				}
			}
		}
	}

}

TEST_F(AttentionTest, Backward2){
   // Set up the attention mechanism with 1 head and a sequence length of 3
    size_t num_heads = 1;
    Attention* att = new Attention(num_heads, T);

	B = 1;
	T = 3;
	C = 3;

    // Set up the input node with shape (1, 3, 3)
    delete[] in->act;
	delete[] in->act_grads;
	in->act = new float[B*T*3*C];
	in->act_grads = new float[B*T*3*C];
	in->shape = {B, T, 3*C};
	in->size = B*T*3*C;

    // Set up the output node with shape (1, 3, 1)
    delete[] out->act;
	delete[] out->act_grads;
	out->act = new float[B*T*C];
	out->act_grads = new float[B*T*C];
	out->shape = {B, T, C};
	out->size = B*T*C;

    // Fill the query, key, and value vectors with easy-to-check values
    // Query vector: [1, 2, 3], [4, 5, 6], [7, 8, 9]
    // Key vector: [10, 11, 12], [13, 14, 15], [16, 17, 18]
    // Value vector: [19, 20, 21], [22, 23, 24], [25, 26, 27]
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < 3; i++) {
            in->act[t * 9 + i] = t * 3 + i + 1; // query
            in->act[t * 9 + 3 + i] = t * 3 + i + 10; // key
            in->act[t * 9 + 6 + i] = t * 3 + i + 19; // value
        }
    }

    // Call the forward pass to fill the buffers
    att->forward(out, in);

    // Fill the output gradients with ones
    for (int i = 0; i < out->size; i++) {
        out->act_grads[i] = 1.0f;
    }

    // Call the backward pass to compute the gradients
    att->backward(out, in);

    // Check the gradients of the query, key, and value vectors
    // dbuffer1 and dbuffer2 should be checked for correct values
    int half_buffer_size = num_heads*B*lrint(T*(T+1)/2);
	int head_size = C / att->num_heads;

    float* dbuffer = att->dbuffer;
    float* dbuffer1 = dbuffer;
    float* dbuffer2 = dbuffer + half_buffer_size;

	float* buffer = att->buffer;
	float* buffer1 = buffer;
	float* buffer2 = buffer + half_buffer_size;

    // Check dbuffer2
    // dbuffer2 should be the output gradients multiplied by the value vector
    for (int t = 0; t < T; t++) {
        for (int t2 = 0; t2 <= t; t2++) {
            float expected_dbuffer2 = 0;
            for (int i = 0; i < head_size; i++) {
                expected_dbuffer2 += out->act_grads[t] * in->act[t2 * 9 + 6 + i];
            }
            EXPECT_FLOAT_EQ(dbuffer2[t * (t + 1) / 2 + t2], expected_dbuffer2);
        }
    }

    // Check dbuffer1
    // dbuffer1 should be the softmax gradients multiplied by the output gradients and the key vector
    for (int t = 0; t < T; t++) {
        for (int t2 = 0; t2 <= t; t2++) {
            float expected_dbuffer1 = 0;
            for (int t3 = 0; t3 <= t; t3++) {
				float local_deriv = buffer2[t * (t+1) / 2 + t3] * ((t2==t3 ? 1.0f : 0.0f) - buffer2[t * (t+1) / 2 + t2]);
                expected_dbuffer1 += dbuffer2[t * (t + 1) / 2 + t3] * local_deriv;
            }
            EXPECT_FLOAT_EQ(dbuffer1[t * (t + 1) / 2 + t2], expected_dbuffer1);
        }
    }
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