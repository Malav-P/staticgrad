#include <gtest/gtest.h>
#include "tokenizer.hpp"
#include "test_common.hpp"

class TokenizerTest : public ::testing::Test {
    protected:
        std::string filename;
        void SetUp() override {
            filename = "/Users/malavpatel/Coding_Projects/StaticGrad/tokenizer/gpt2_vocab.bin";

        }

        void TearDown() override {

        }

};


TEST_F(TokenizerTest, Constructor) {

    Tokenizer tk = Tokenizer(filename);

}

TEST_F(TokenizerTest, Decode) {

    Tokenizer tk = Tokenizer(filename);

    int tokenIDs[2] = {0, 1};
    int length = 2;

    std::string decoded = tk.decode(tokenIDs, length);
    std::string expected = "!\"";

    EXPECT_TRUE(decoded == expected);

    // use std::vector container for tokens. this one is for "hello world"
    std::vector<int> tokenids{31373, 995};

    decoded = tk.decode(tokenids);
    EXPECT_TRUE(decoded == "hello world");
}

TEST_F(TokenizerTest, Encode) {
    Tokenizer tk = Tokenizer(filename);

    std::string str = "hello world";
    std::vector<int> tokenIDs = tk.encode(str);

    EXPECT_GT(tokenIDs.size(), 0);

    std::cout << "encoded: ";
    for (auto token : tokenIDs){
        std::cout << token << ", ";
    }
    std::cout << std::endl;

    std::string decoded = tk.decode(tokenIDs);
    std::cout << "decoded: " <<  decoded << std::endl;
}

TEST_F(TokenizerTest, EdgeCases) {

    Tokenizer tk = Tokenizer(filename);

    int tokenIDs[1] = {-1};
    int length = 1;
    EXPECT_THROW(tk.decode(tokenIDs, length), std::out_of_range);

    int* invalid_array = nullptr;
    length = 1;
    EXPECT_THROW(tk.decode(invalid_array, length), std::invalid_argument);

}

TEST_F(TokenizerTest, EncodeDecodeTest) {

    Tokenizer* tokenizer = new Tokenizer(filename);

    std::vector<std::string> testStrings = {
        "hello world",
        "this is a test",
        "tokenizer test",
        "", // Empty string
        "a", // Single character
        "abcdefghijklmnopqrstuvwxyz", // Long string
        "1234567890", // Numbers
        "!@#$%^&*()", // Special characters
    };

    for (const auto& str : testStrings) {
        std::vector<int> encoded = tokenizer->encode(str);
        std::string decoded = tokenizer->decode(encoded);
        EXPECT_EQ(str, decoded);
    }

    delete tokenizer;
}

// Main function to run the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}