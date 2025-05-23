#include <gtest/gtest.h>
#include "classes.hpp"
#include "test_common.hpp"


class AddTest : public ::testing::Test {
 protected:
  void SetUp() override {
    B = 1;
    T = 1;
    C = 768;

    in = new Node();
    setup_node(in, {B, T, C});

    out = new Node();
    setup_node(out, {B, T, C});

  }

  void TearDown() override {
    teardown_node(in);
    teardown_node(out);
  }

  size_t B;
  size_t T;
  size_t C;
  Node* in;
  Node* out;
};



TEST_F(AddTest, ToArrayofOne) {
  Add* add = new Add();

  fillArrayWithRandom(in->act, in->size);
  fillArrayWithOnes(out->act, out->size);

  add->forward(out, in);

  for (size_t i = 0; i < out->size; i++){
    EXPECT_EQ(out->act[i], 1.0f + in->act[i]);
  }

  delete add;
}

TEST_F(AddTest, ToArrayofRandom) {
  Add* add = new Add();

  fillArrayWithRandom(in->act, in->size);
  fillArrayWithRandom(out->act, out->size);

  float* expected = new float[out->size];
  for (size_t i = 0; i < out->size; i++) {
    expected[i] = in->act[i] + out->act[i];
  }

  add->forward(out, in);

  for (size_t i = 0; i < out->size; i++){
    EXPECT_EQ(out->act[i], expected[i]);
  }

  delete[] expected;
  delete add;
}

TEST_F(AddTest, ToArrayofZeros) {
	Add* add = new Add();

	fillArrayWithRandom(in->act, in->size);
  std::memset(out->act, 0, out->size * sizeof(float)); // fill array with zeros

	add->forward(out, in);

	for (size_t i = 0; i < out->size; i++){
	EXPECT_EQ(out->act[i], in->act[i]);
	}

	delete add;
}


TEST_F(AddTest, Backward){
	Add* add = new Add();

	fillArrayWithRandom(out->act_grads, out->size);
	std::memset(in->act_grads, 0, in->size * sizeof(float));

	add->backward(out, in);

	for (size_t i = 0; i < in->size; i++){
		EXPECT_FLOAT_EQ(in->act_grads[i], out->act_grads[i]);
	}

	add->backward(out, in);

	for (size_t i = 0; i < in->size; i++){
		EXPECT_FLOAT_EQ(in->act_grads[i], 2*out->act_grads[i]);
	}
}

TEST_F(AddTest, InPlaceBackward){
	Add* add = new Add();

	fillArrayWithRandom(out->act_grads, out->size);

	float* buffer = new float[out->size];
	std::memcpy(buffer, out->act_grads, out->size * sizeof(float));

	add->backward(out, out);

	for (size_t i = 0; i < in->size; i++){
		EXPECT_FLOAT_EQ(out->act_grads[i], 2*buffer[i]);
	}

	delete[] buffer;

}

TEST_F(AddTest, InPlaceForward){
	Add* add = new Add();

	fillArrayWithRandom(out->act, out->size);

	float* buffer = new float[out->size];
	std::memcpy(buffer, out->act, out->size * sizeof(float));

	add->forward(out, out);

	for (size_t i = 0; i < in->size; i++){
		EXPECT_FLOAT_EQ(out->act[i], 2*buffer[i]);
	}

	delete[] buffer;

}



// Main function to run the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}