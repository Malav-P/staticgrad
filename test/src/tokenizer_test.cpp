#include <gtest/gtest.h>
#include "tokenizer.hpp"
#include "test_common.hpp"

class TokenizerTest : public ::testing::Test {
    protected:
        std::string filename;
        void SetUp() override {
            filename = "/Users/malavpatel/Coding_Projects/StaticGrad/tokenizer/gpt2_vocab.bin";
        }
        void TearDown() override {}
};


TEST_F(TokenizerTest, Constructor) {

    Tokenizer tk = Tokenizer(filename);

}

TEST_F(TokenizerTest, Decode) {

    Tokenizer tk = Tokenizer(filename);

    u_int16_t tokenIDs[2] = {0, 1};
    int length = 2;

    std::string decoded = tk.decode(tokenIDs, length);
    std::string expected = "!\"";

    EXPECT_TRUE(decoded == expected);

    // use std::vector container for tokens. this one is for "hello world"
    std::vector<u_int16_t> tokenids{31373, 995};

    decoded = tk.decode(tokenids);
    EXPECT_TRUE(decoded == "hello world");
}

TEST_F(TokenizerTest, eot) {

    Tokenizer tk = Tokenizer(filename);

    u_int16_t tokenIDs[1] = {50256};
    int length = 1;

    std::string decoded = tk.decode(tokenIDs, length);
    std::string expected = "<|endoftext|>";

    EXPECT_TRUE(decoded == expected);

    // use std::vector container for tokens. this one is for the eot token
    std::vector<u_int16_t> tokenids{50256};

    decoded = tk.decode(tokenids);
    EXPECT_TRUE(decoded == "<|endoftext|>");
}

TEST_F(TokenizerTest, Encode) {
    Tokenizer tk = Tokenizer(filename);

    std::string str = "hello world";
    std::vector<u_int16_t> tokenIDs = tk.encode(str);

    EXPECT_GT(tokenIDs.size(), size_t(0));

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

    u_int16_t* invalid_array = nullptr;
    int length = 1;
    EXPECT_THROW(tk.decode(invalid_array, length), std::invalid_argument);

}

TEST_F(TokenizerTest, EncodeDecodeTest) {

    Tokenizer* tokenizer = new Tokenizer(filename);

    std::vector<std::string> testStrings = {
        "hello world",
        "this is a test with puncuation. I want! to return the same string from the tokenizer",
        "tokenizer test",
        "", // Empty string
        "a", // Single character
        "abcdefghijklmnopqrstuvwxyz", // Long string
        "1234567890", // Numbers
        "!@#$%^&*()", // Special characters
    };

    for (const auto& str : testStrings) {
        std::vector<u_int16_t> encoded = tokenizer->encode(str);
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