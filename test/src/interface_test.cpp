#include <gtest/gtest.h>
#include "interface.hpp"
#include "test_common.hpp"
#include "node.hpp"

class SetupTeardownTest : public ::testing::Test {
	protected:

        void SetUp() override {}

        void TearDown() override {}
};


TEST_F(SetupTeardownTest, NoMemoryLeaks) {
    GPT2* model = nullptr;
    DataStream* ds = nullptr;
    Tokenizer* tk = nullptr;
    Node* out = new Node();
    Node* in = new Node();

    size_t B = 1; // batch size
    size_t T = 1; // sequence length
    size_t C = 768;
    size_t L = 12;
    size_t V = 50257;

    setup(model, ds, tk);
    Activation* activations = new Activation(B, T, C, L, V);
    activations->point_Nodes(out, in);
    EXPECT_NE(model, nullptr);

    tear_down(model, ds, tk);
    delete activations;
    delete out;
    delete in;
    
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