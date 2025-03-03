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
    setup_node(in, {B, T, V});

    out = new Node();
    setup_node(out, {B, T, V});
  }

  void TearDown() override {
    teardown_node(out);
    teardown_node(in);
  }

  size_t B;
  size_t T;
  size_t V;
  
  Node* in;
  Node* out;
};


TEST_F(UtilsTest, sample_token) {
    float logits[] = {1, 2, 3, 4};
    float probabilities[] = {0.032059, 0.087144, 0.236883, 0.643914};
    size_t length = sizeof(probabilities) / sizeof(float);

    uint16_t token;

    EXPECT_NO_THROW(token = sample_token(logits, length, true));

    int* counts = new int[length]();
    for (int i = 0; i < 10000; i++){
        token = sample_token(logits, 4, true);
        counts[token] += 1;
    }

    for (size_t i = 0; i < length; i++){
        EXPECT_NEAR(counts[i], 10000*probabilities[i], 200);
    }

    delete[] counts;


    // test non-random portion
    EXPECT_EQ(sample_token(probabilities, length, false), 3);

}






// Main function to run the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}