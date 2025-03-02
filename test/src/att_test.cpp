#include <gtest/gtest.h>
#include "classes.hpp"
#include "test_common.hpp"


class AttentionTest : public ::testing::Test {
    protected:
     void SetUp() override {
       // Default values, can be overridden in test cases
       SetDimensions(2, 3, 768);
     }
   
     void TearDown() override {
       teardown_node(out);
       teardown_node(in);
     }
   
     void SetDimensions(size_t b, size_t t, size_t c) {
       B = b;
       T = t;
       C = c;
       maxT = 1024;
   
       in = new Node();
       setup_node(in, {B, T, 3*C});
   
       out = new Node();
       setup_node(out, {B, T, C});
     }
   
     size_t B;
     size_t T;
     size_t C;
     size_t maxT;
   
     Node* in;
     Node* out;
   };
   
TEST_F(AttentionTest, CustomDimensionsTest) {
    SetDimensions(4, 5, 512); // Custom dimensions for this test
    
    EXPECT_EQ(B, 4);
    EXPECT_EQ(T, 5);
    EXPECT_EQ(C, 512);
}

TEST_F(AttentionTest, Forward2) {
    SetDimensions(1, 2, 4);

    Attention* att = new Attention(1);

	// fill query, key, value
	for (size_t b = 0; b < B; b++){
		for (size_t t = 0; t < T; t++){
			for (size_t i = 0; i < C; i++){
				// fill query
				in->act[b*T*3*C + t*3*C + i] = 1.0f / C * (i+1) * (t+1);

				// fill key
				in->act[b*T*3*C + t*3*C + C + i] = 2.0f / C * (i+1) * (t+1);

				// fill value
				in->act[b*T*3*C + t*3*C + C + C + i] = 3.0f / C;
			}
		}
	}

    att->forward(out, in);

    // test raw attention scores
    float expected[3] = {15.0f / 8.0f , 15.0f / 4.0f, 15.0f / 2.0f};
    
    float* pre_softmax_bth = att->buffer;

    for (size_t i=0; i < 3; i++){
        EXPECT_FLOAT_EQ(expected[i], pre_softmax_bth[i]);
    }

    delete att;
}

TEST_F(AttentionTest, Forward) {
	size_t num_heads = 12;
	Attention* att = new Attention(num_heads);

	int half_buffer_size = num_heads*B*lrint(T*(T+1)/2);
	size_t head_size = C / att->num_heads;


	// fill query, key, value
	for (size_t b = 0; b < B; b++){
		for (size_t t = 0; t < T; t++){
			for (size_t i = 0; i < C; i++){
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

	for (size_t b = 0; b < B; b++){
		for (size_t t = 0; t < T; t++){
			for (size_t nh = 0; nh < att->num_heads; nh++){

				float* buffer2 = att->buffer + half_buffer_size + b*num_heads*lrint(T*(T+1)/2) + lrint(t*(t+1)/2)*num_heads + (t + 1)*nh;
				tracker += (t + 1);

				float sum = 0.0f;
				for (size_t t2 = 0; t2 <= t; t2++){
					sum += buffer2[t2];
				}

				EXPECT_NEAR(sum, 1.0f, 1e-5);

                // get output vector
                float* out_act = out->act + b * T * C + t * C + nh * head_size;

				float* expected = new float[head_size]();
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


TEST_F(AttentionTest, Backward){

	// ----- to fill buffers

	size_t num_heads = 12;
	Attention* att = new Attention(num_heads);

	int head_size = C / att->num_heads;

	fillArrayWithRandom(in->act, in->size);

	// calling att->backward before att->forward should throw
	EXPECT_THROW(att->backward(out, in), std::runtime_error);

	att->forward(out, in);


	// ------- Begin testing backwards ---- 

	fillArrayWithOnes(out->act_grads, out->size);
	memset(in->act_grads, 0, in->size * sizeof(float));

	for (size_t i = 0 ; i < in->size; i++){
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
    Attention* att = new Attention(num_heads);

	B = 1;
	T = 3;
	C = 3;

    setup_node(out, {B, T, C});
    setup_node(in, {B, T, 3*C});

    // Fill the query, key, and value vectors with easy-to-check values
    // Query vector: [1, 2, 3], [4, 5, 6], [7, 8, 9]
    // Key vector: [10, 11, 12], [13, 14, 15], [16, 17, 18]
    // Value vector: [19, 20, 21], [22, 23, 24], [25, 26, 27]
    for (size_t t = 0; t < T; t++) {
        for (size_t i = 0; i < 3; i++) {
            in->act[t * 9 + i] = t * 3 + i + 1; // query
            in->act[t * 9 + 3 + i] = t * 3 + i + 10; // key
            in->act[t * 9 + 6 + i] = t * 3 + i + 19; // value
        }
    }

    // Call the forward pass to fill the buffers
    att->forward(out, in);

    // Fill the output gradients with ones
    for (size_t i = 0; i < out->size; i++) {
        out->act_grads[i] = 1.0f;
    }

    // Call the backward pass to compute the gradients
    att->backward(out, in);

    // Check the gradients of the query, key, and value vectors
    // dbuffer1 and dbuffer2 should be checked for correct values
    int half_buffer_size = num_heads*B*lrint(T*(T+1)/2);
	size_t head_size = C / att->num_heads;

    float* dbuffer = att->dbuffer;
    float* dbuffer1 = dbuffer;
    float* dbuffer2 = dbuffer + half_buffer_size;

	float* buffer = att->buffer;
	float* buffer2 = buffer + half_buffer_size;

    // Check dbuffer2
    // dbuffer2 should be the output gradients multiplied by the value vector
    for (size_t t = 0; t < T; t++) {
        for (size_t t2 = 0; t2 <= t; t2++) {
            float expected_dbuffer2 = 0;
            for (size_t i = 0; i < head_size; i++) {
                expected_dbuffer2 += out->act_grads[t] * in->act[t2 * 9 + 6 + i];
            }
            EXPECT_FLOAT_EQ(dbuffer2[t * (t + 1) / 2 + t2], expected_dbuffer2);
        }
    }

    // Check dbuffer1
    // dbuffer1 should be the softmax gradients multiplied by the output gradients and the key vector
    for (size_t t = 0; t < T; t++) {
        for (size_t t2 = 0; t2 <= t; t2++) {
            float expected_dbuffer1 = 0;
            for (size_t t3 = 0; t3 <= t; t3++) {
				float local_deriv = buffer2[t * (t+1) / 2 + t3] * ((t2==t3 ? 1.0f : 0.0f) - buffer2[t * (t+1) / 2 + t2]);
                expected_dbuffer1 += dbuffer2[t * (t + 1) / 2 + t3] * local_deriv;
            }
            EXPECT_FLOAT_EQ(dbuffer1[t * (t + 1) / 2 + t2], expected_dbuffer1);
        }
    }
}


TEST_F(AttentionTest, AllOnes) {
  size_t num_heads = 12;
  Attention* att = new Attention(num_heads);

  fillArrayWithOnes(in->act, in->size);

  att->forward(out, in);

  for (size_t i = 0; i < out->size; i++){
    EXPECT_EQ(out->act[i], 1.0);
  }

  delete att;
}

TEST_F(AttentionTest, InvalidShapes) {
  size_t num_heads = 12;
  Attention* att = new Attention(num_heads);

  in->size = B*T*C;
  EXPECT_THROW(att->forward(out, in), std::invalid_argument);

  delete att;
}



int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}