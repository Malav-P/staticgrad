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

TEST_F(TokenizerTest, FirstTwoToken) {

    std::string filename = "/Users/malavpatel/Coding_Projects/StaticGrad/tokenizer/gpt2_vocab.bin";

    Tokenizer tk = Tokenizer(filename);

    int tokenIDs[2] = {0, 1};
    int length = 2;

    std::string decoded = tk.decodeSequence(tokenIDs, length);
    std::string expected = "!\"";

    EXPECT_TRUE(decoded == expected);
}

TEST_F(TokenizerTest, EdgeCases) {

    std::string filename = "/Users/malavpatel/Coding_Projects/StaticGrad/tokenizer/gpt2_vocab.bin";

    Tokenizer tk = Tokenizer(filename);

    int tokenIDs[1] = {-1};
    int length = 1;
    EXPECT_THROW(tk.decodeSequence(tokenIDs, length), std::out_of_range);

    int* invalid_array = nullptr;
    length = 1;
    EXPECT_THROW(tk.decodeSequence(invalid_array, length), std::invalid_argument);



}

// Main function to run the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}