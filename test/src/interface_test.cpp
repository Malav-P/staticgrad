#include <gtest/gtest.h>
#include <fstream>
#include "interface.hpp"
#include "test_common.hpp"
#include "node.hpp"
#include "gpt2.hpp"

const std::string PREFIX = REPO_PREFIX;

class SetupTeardownTest : public ::testing::Test {
	protected:

        void SetUp() override {}

        void TearDown() override {}
};


TEST_F(SetupTeardownTest, NoMemoryLeaks) {
    GPT2* model = nullptr;
    DataStream* ds = nullptr;
    Tokenizer* tk = nullptr;
    Activation* activations = nullptr;
    Node* out = nullptr;
    Node* in = nullptr;
    bool pretrained = true;

    size_t B = 1; // batch size
    size_t T = 1; // sequence length

    setup(model, ds, tk, activations, out, in, B, T, pretrained);
    EXPECT_NE(model, nullptr);

    tear_down(model, ds, tk, activations, out, in);
    
    EXPECT_EQ(model, nullptr);

}

TEST_F(SetupTeardownTest, helloworld) {

    // setup model pointers and data structures
    GPT2* model = nullptr;
    DataStream* ds = nullptr;
    Tokenizer* tk = nullptr;
    Activation* activations = nullptr;
    Node* out = nullptr;
    Node* in = nullptr;
    size_t B = 1;
    size_t T = 3;
    size_t V = 50257;
    bool pretrained = true;

    setup_model(model, pretrained);
    setup_tokenizer(tk);
    setup_activations(activations, out, in, B, T, model);

    in->act[0] = 31373;
    in->act[1] = 995;

    in->current_T = 2;

    model->forward(out, in);
    float* logits = out->act - 2*(V);
    std::ifstream file(PREFIX + "bin/hello_world_logits.bin", std::ios::binary);

    if (!file) {
        std::cerr << "Error opening file!\n";
    }

    // Determine file size
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Allocate memory for float data
    size_t num_floats = size / sizeof(float);
    float* data = new float[num_floats];

    // Read binary data into the allocated memory
    if (file.read(reinterpret_cast<char*>(data), size)) {
        for (size_t i = 0; i < 3; i++){
            EXPECT_NEAR(logits[i], data[i], 1e-3);
        }
        
    } else {
        std::cerr << "Error reading file!\n";
    }



    // Cleanup
    delete[] data;
    file.close();


    // deallocate memory
    tear_down(model, ds, tk, activations, out, in);

}

int main(int argc, char **argv) {


    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}