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
    Optimizer* opt = nullptr;
    DataStream* ds = nullptr;
    Tokenizer* tk = nullptr;
    Activation* activations = nullptr;
    Node* out = nullptr;
    Node* in = nullptr;
    bool pretrained = true;

    size_t B = 1; // batch size
    size_t T = 1; // sequence length

    setup(model, opt, ds, tk, activations, out, in, B, T, pretrained);
    EXPECT_NE(model, nullptr);

    tear_down(model, opt, ds, tk, activations, out, in);
    
    EXPECT_EQ(model, nullptr);

}

TEST_F(SetupTeardownTest, helloworld) {
    std::ifstream file(PREFIX + "bin/hello_world_logits.bin", std::ios::binary);
    if (!file.good()) {
        std::cerr << "\033[1;33mWARN\033[0m: File "<< PREFIX + "bin/hello_world_logits.bin"<< " not found." << std::endl;
        GTEST_SKIP();
    }
    // setup model pointers and data structures
    GPT2* model = nullptr;
    Tokenizer* tk = nullptr;
    Activation* activations = nullptr;
    Node* out = nullptr;
    Node* in = nullptr;
    size_t B = 1;
    size_t T = 3;
    bool pretrained = true;

    setup_model(model, pretrained);
    setup_tokenizer(tk);
    setup_activations(activations, out, in, B, T, model);

    in->act[0] = 31373;
    in->act[1] = 995;
    in->current_T = 1;

    model->forward(out, in);
    in->current_T = 2;
    model->inference_time_opt();
    model->forward(out, in);
    float* logits = out->act + (model->V);

    // Determine file size
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    // Allocate memory for float data
    size_t num_floats = size / sizeof(float);
    float* data = new float[num_floats];
    // Read binary data into the allocated memory
    if (file.read(reinterpret_cast<char*>(data), size)) {
        for (size_t i = 0; i < model->V; i++){
            EXPECT_NEAR(logits[i], data[i], 1e-3);
        }
        
    } else {
        std::cerr << "Error reading file!" << std::endl;
    }

    // Cleanup
    delete[] data;
    file.close();

    // deallocate memory
    teardown_activations(activations, out, in);
    teardown_tokenizer(tk);
    teardown_model(model);

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}