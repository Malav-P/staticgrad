#include <gtest/gtest.h>
#include "interface.hpp"
#include "test_common.hpp"

class SetupTeardownTest : public ::testing::Test {
	protected:

        void SetUp() override {}

        void TearDown() override {}
};


TEST_F(SetupTeardownTest, NoMemoryLeaks) {
    GPT2* model = nullptr;
    DataStream* ds = nullptr;
    Tokenizer* tk = nullptr;
    Node* out = nullptr;
    Node* in = nullptr;

    size_t B = 1; // batch size
    size_t T = 1; // sequence length

    setup(model, ds, tk, out, in, B, T);
    EXPECT_NE(model, nullptr);

    tear_down(model, ds, tk, out, in);
    EXPECT_EQ(model, nullptr);

}

int main(int argc, char **argv) {

    // GPT2* model = nullptr;
    // DataStream* ds = nullptr;
    // Tokenizer* tk = nullptr;
    // Node* out = nullptr;
    // Node* in = nullptr;

    // size_t B = 1; // batch size
    // size_t T = 1; // sequence length

    // setup(model, ds, tk, out, in, B, T);
    // tear_down(model, ds, tk, out, in);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}