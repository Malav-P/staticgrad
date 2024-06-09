#include <gtest/gtest.h>
#include "tokenizer.hpp"
#include "test_common.hpp"

class TokenizerTest : public ::testing::Test {
    protected:
        void SetUp() override {

        }

        void TearDown() override {

        }

};


TEST_F(TokenizerTest, Constructor) {

    std::string filename = "/Users/malavpatel/Coding_Projects/StaticGrad/tokenizer/gpt2_vocab.bin";

    Tokenizer tk = Tokenizer(filename);

}


// Main function to run the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}