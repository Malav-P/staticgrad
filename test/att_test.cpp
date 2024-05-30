#include <iostream>
#include <memory>
#include <gtest/gtest.h>
#include "src/classes.hpp"

using namespace std;

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

	// calling att->backward before att->forward should throw
	EXPECT_THROW(att->backward(out, in), std::runtime_error);

	att->forward(out, in);


	// ------- Begin testing backwards ---- 

	att->backward(out, in);

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