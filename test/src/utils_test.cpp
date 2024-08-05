#include <gtest/gtest.h>
#include "utils.hpp"
#include "test_common.hpp"

class UtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    B = 1;
    T = 1;
    V = 10;

    in = new Node();
    in->act = new float[B * T * V];
    in->act_grads = new float[B * T * V];
    in->shape = {B, T, V};
    in->size = B * T * V;

    out = new Node();
    out->act = new float[B * T * V];
    out->act_grads = new float[B * T * V];
    out->shape = {B, T, V};
    out->size = B * T * V;
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
  size_t V;
  Node* in;
  Node* out;
};

/**
 * @brief Test that the crossentropy_forward function can compute the cross-entropy loss successfully.
 *
 * This test sets up a scenario where the input node contains a softmax output, and the target node contains the true labels.
 * It then calls the crossentropy_forward function and verifies that the output node's activation is correctly computed as the cross-entropy loss.
 */
TEST_F(UtilsTest, crossentropyforward) {

    u_int16_t targets[] = {8}; // token ID of the target. We only have one because B = T = 1 for the test
    float softmax_out[] = {0.12, 0.03, 0.25, 0.07, 0.08, 0.15, 0.04, 0.1, 0.09, 0.07};
    for (size_t i = 0 ; i < V; i++){
        in->act[i] = softmax_out[i];
    }

    EXPECT_NO_THROW(crossentropy_forward(out, in, targets));

    EXPECT_FLOAT_EQ(out->act[0], -std::logf(softmax_out[targets[0]]));
}

/**
 * @brief Test that the crossentropy_softmax_backward function can compute the gradients of the cross-entropy loss with respect to the input node's activations successfully.
 *
 * This test sets up a scenario where the output node contains a softmax output, and the target node contains the true labels.
 * It then calls the crossentropy_softmax_backward function and verifies that the input node's activation gradients are correctly computed.
 */
TEST_F(UtilsTest, crossentropy_softmax_backward) {

    u_int16_t targets[] = {8}; // token ID of the target. We only have one because B = T = 1 for the test
    float softmax_out[] = {0.12, 0.03, 0.25, 0.07, 0.08, 0.15, 0.04, 0.1, 0.09, 0.07};
    for (size_t i = 0 ; i < V; i++){
        out->act[i] = softmax_out[i];
    }

    float temp = 1.0f;

    EXPECT_NO_THROW(crossentropy_softmax_backward(out, in, targets, temp));

    for(size_t i = 0; i < V; i++){
        float indicator = (i == targets[0]) ? 1.0f : 0.0f;
        float expected = (softmax_out[i] - indicator) / (B*T) / temp;
        EXPECT_FLOAT_EQ(in->act_grads[i], expected);
    }
    
}

TEST_F(UtilsTest, InvalidInputTest) {
    float probabilities[] = {0.1, 0.2, -0.3, 0.4};
    size_t length = sizeof(probabilities) / sizeof(probabilities[0]);

    // Check that the function throws an exception when given invalid input
    EXPECT_THROW(sample_token(probabilities, length), std::invalid_argument);
}

TEST_F(UtilsTest, InvalidSum) {
    float probabilities[] = {0.1, 0.2, 0.3, 0.3};
    size_t length = sizeof(probabilities) / sizeof(probabilities[0]);

    // Check that the function throws an exception when given invalid input
    EXPECT_THROW(sample_token(probabilities, length), std::invalid_argument);
}

TEST_F(UtilsTest, sample_token) {
    float probabilities[] = {0.4, 0.1, 0.2, 0.3};
    size_t length = sizeof(probabilities) / sizeof(float);

    u_int16_t token = 0;

    EXPECT_NO_THROW(token = sample_token(probabilities, length));

    int* counts = new int[length];
    for (int i = 0; i < 10000; i++){
        u_int16_t token = sample_token(probabilities, 4, true);
        
        counts[token] += 1;
        
    }

    for (int i = 0; i < length; i++){
        EXPECT_NEAR(counts[i], 10000*probabilities[i], 200);
    }

    delete[] counts;


    // test non-random portion
    EXPECT_EQ(sample_token(probabilities, length), 0);

}

/**
 * @brief Test that the load_weights function can load model weights from a file.
 *
 * This test verifies that the load_weights function works correctly and that
 * it can load model weights from a file. It also checks that the function throws
 * a std::runtime_error when the expected number of bytes does not match the
 * actual number of bytes in the file.
 */
TEST_F(UtilsTest, load_weights) {

    std::string filepath = "/Users/malavpatel/Coding_Projects/StaticGrad/models/gpt2_weights.bin";

    int num_params = 124439808;
    auto params = new float[num_params]; // num parameters in gpt2 is 124439808


    EXPECT_NO_THROW(load_weights(params, filepath));

    int wrong_expected_bytes = 10;
    EXPECT_THROW(load_weights(params, filepath, wrong_expected_bytes), std::runtime_error);

    int correct_expected_bytes = num_params*sizeof(float);
    EXPECT_NO_THROW(load_weights(params, filepath, correct_expected_bytes));

    delete[] params;

}




// Main function to run the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}