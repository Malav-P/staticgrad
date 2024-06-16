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

TEST_F(TokenizerTest, Decode) {

    std::string filename = "/Users/malavpatel/Coding_Projects/StaticGrad/tokenizer/gpt2_vocab.bin";

    Tokenizer tk = Tokenizer(filename);

    int tokenIDs[2] = {0, 1};
    int length = 2;

    std::string decoded = tk.decodeSequence(tokenIDs, length);
    std::string expected = "!\"";

    EXPECT_TRUE(decoded == expected);

    // use std::vector container for tokens. this one is for "hello world"
    std::vector<int> tokenids{31373, 995};

    decoded = tk.decodeSequence(tokenids);
    EXPECT_TRUE(decoded == "hello world");
}

TEST_F(TokenizerTest, Encode) {
    std::string filename = "/Users/malavpatel/Coding_Projects/StaticGrad/tokenizer/gpt2_vocab.bin";
    Tokenizer tk = Tokenizer(filename);

    std::string str = "hello world";
    std::vector<int> tokenIDs = tk.encode(str);

    EXPECT_GT(tokenIDs.size(), 0);

    std::cout << "encoded: ";
    for (auto token : tokenIDs){
        std::cout << token << ", ";
    }
    std::cout << std::endl;

    std::string decoded = tk.decodeSequence(tokenIDs);
    std::cout << "decoded: " <<  decoded << std::endl;
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