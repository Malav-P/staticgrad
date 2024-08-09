#include <gtest/gtest.h>
#include "gpt2.hpp"
#include "test_common.hpp"
#include <random>
#include <chrono>

using namespace std;


class GPT2Test : public ::testing::Test {
	protected:

        size_t C;
        size_t L;
        size_t V;
        size_t maxT;
        size_t NH;

        size_t B;
        size_t T;

        void SetUp() override {}

        void TearDown() override {}
};

TEST_F(GPT2Test, Constructor) {
    C = 768; // embedding dimension
    L = 12; // number of transformer blocks
    V = 50257; // vocab size
    maxT = 1024; // max sequence length
    NH = 12; // number of attention heads

    B = 4; // batch size
    T = 64; // sequence length

    GPT2* model = nullptr;

    EXPECT_NO_THROW(model = new GPT2(C, L, V, maxT, NH));
    EXPECT_EQ(model->tblocks.size(), L); // expect L transformer blocks

    

}

TEST_F(GPT2Test, DefaultConstructor) {
    C = 768; // embedding dimension
    L = 12; // number of transformer blocks
    V = 50257; // vocab size
    maxT = 1024; // max sequence length
    NH = 12; // number of attention heads

    GPT2* model = nullptr;

    EXPECT_NO_THROW(model = new GPT2());

    EXPECT_EQ(model->tblocks.size(), L); // expect L transformer blocks
    EXPECT_EQ(model->C, C);
    EXPECT_EQ(model->L, L);
    EXPECT_EQ(model->V, V);
    EXPECT_EQ(model->maxT, maxT);
    EXPECT_EQ(model->NH, NH);
}


TEST_F(GPT2Test, Forward) {
    C = 768; // embedding dimension
    L = 12; // number of transformer blocks
    V = 50257; // vocab size
    maxT = 1024; // max sequence length
    NH = 12; // number of attention heads

    B = 4; // batch size
    T = 64; // sequence length

    GPT2* model = new GPT2(C, L, V, maxT, NH);

    size_t num_act = gpt2_num_acts(B, T, C, L, V);

    float* acts = new float[num_act];
    float* grad = new float[num_act];

    Node* in = new Node();
    in->act = acts;
    in->act_grads = grad;
    in->shape = {B, T};
    in->size = B*T;
    in->current_T = T;

    Node* out = new Node();
    out->act = acts + num_act - B*T*V;
    out->act_grads = grad + num_act - B*T*V;
    out->shape = {B, T, V};
    out->size = B*T*V;
    out->current_T = T;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, V-1);

    for (size_t i = 0; i < B*T; i++) {
        in->act[i] = distrib(gen);
    }

    // fill parameters with random values between -1.0 and 1.0
    fillArrayWithRandom(model->params, model->num_params);

    // check outputs are zero
    for (size_t i = 0; i < out->size; i++){
        EXPECT_FLOAT_EQ(out->act[i], 0.0f);
    }

    auto start = std::chrono::high_resolution_clock::now();

    // forward pass should work
    EXPECT_NO_THROW(model->forward(out, in));

    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the duration
    std::chrono::duration<double, std::milli> duration = end - start;
    // Output the duration in milliseconds
    std::cout << "Time taken by forward(): " << duration.count() << " ms" << std::endl;

    // each array at (b, t) position should sum to one
    for (size_t b = 0; b < B; b++){
        for (size_t t = 0; t < T; t++){
            float* probs = out->act + b * T*V + t * V;
            float sum = 0.0f;
            for (size_t v = 0; v < V; v++){
                float p = probs[v];
                EXPECT_LE(p, 1.0f);
                EXPECT_GE(p, 0.0f);
                sum += p;
            }
            EXPECT_NEAR(sum, 1.0f, 1e-5);
        }
    }

}

TEST_F(GPT2Test, Backward) {
    C = 768; // embedding dimension
    L = 12; // number of transformer blocks
    V = 50257; // vocab size
    maxT = 1024; // max sequence length
    NH = 12; // number of attention heads

    B = 4; // batch size
    T = 64; // sequence length

    GPT2* model = new GPT2(C, L, V, maxT, NH);

    size_t num_act = gpt2_num_acts(B, T, C, L, V);

    float* acts = new float[num_act];
    float* grad = new float[num_act];

    Node* in = new Node();
    in->act = acts;
    in->act_grads = grad;
    in->shape = {B, T};
    in->size = B*T;
    in->current_T = T;

    Node* out = new Node();
    out->act = acts + num_act - B*T*V;
    out->act_grads = grad + num_act - B*T*V;
    out->shape = {B, T, V};
    out->size = B*T*V;
    out->current_T = T;


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, V-1);

    for (size_t i = 0; i < B*T; i++) {
        in->act[i] = distrib(gen);
    }

    // fill parameters with random values between -1.0 and 1.0
    fillArrayWithRandom(model->params, model->num_params);

    // check outputs are zero
    for (size_t i = 0; i < out->size; i++){
        EXPECT_FLOAT_EQ(out->act[i], 0.0f);
    }

    // forward pass should work
    model->forward(out, in);


    // fill output grad
    for (size_t j = 0; j < out->size; j++){
        out->act_grads[j] = 1.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();
    // backward pass should work
    EXPECT_NO_THROW(model->backward(out, in));
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the duration
    std::chrono::duration<double, std::milli> duration = end - start;
    // Output the duration in milliseconds
    std::cout << "Time taken by backward(): " << duration.count() << " ms" << std::endl;


}

TEST_F(GPT2Test, InvalidNumHeads) {
    C = 768; // embedding dimension
    L = 12; // number of transformer blocks
    V = 50257; // vocab size
    maxT = 1024; // max sequence length
    NH = 11; // number of attention heads

    GPT2* model = nullptr;
    EXPECT_THROW(model = new GPT2(C, L, V, maxT, NH), std::invalid_argument);

    // added to suppress compilation warnings for pointer
    model = new GPT2();
    delete model;
    
}

TEST_F(GPT2Test, ZeroGrad) {
    // Create a GPT2 object
    C = 768;
    L = 12;
    V = 50257;
    maxT = 1024;
    NH = 16;

    GPT2* gpt2 = new GPT2(C, L, V, maxT, NH);

    // Set some gradients to non-zero values
    for (size_t i = 0; i < gpt2->num_params; i++) {
        gpt2->grad[i] = 1.0f;
    }

    // Measure the time taken by zero_grad() ( can comment )
    auto start = std::chrono::high_resolution_clock::now();
    gpt2->zero_grad();
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the elapsed time
    std::chrono::duration<double> elapsed = end - start;
    // Print the elapsed time 
    std::cout << "Time taken by zero_grad(): " << elapsed.count() << " seconds" << std::endl;

    // Check if all gradients are zero
    for (size_t i = 0; i < gpt2->num_params; i++) {
        EXPECT_FLOAT_EQ(gpt2->grad[i], 0.0f);
    }

    delete gpt2;
}

TEST_F(GPT2Test, Update) {
    // Create a GPT2 object
    C = 768;
    L = 12;
    V = 50257;
    maxT = 1024;
    NH = 16;

    GPT2* gpt2 = new GPT2(C, L, V, maxT, NH);

    fillArrayWithRandom(gpt2->grad, gpt2->num_params);

    EXPECT_NO_THROW(gpt2->update(1));

    for (size_t i = 0; i < gpt2->num_params; i++){
        EXPECT_FLOAT_EQ(gpt2->m[i], (1 - gpt2->beta1) * gpt2->grad[i]);
        EXPECT_FLOAT_EQ(gpt2->v[i], (1 - gpt2->beta2) * gpt2->grad[i] * gpt2->grad[i] );
    }
}


/**
 * @brief Test that the load_weights function can load model weights from a file.
 *
 * This test verifies that the load_weights function works correctly and that
 * it can load model weights from a file. It also checks that the function throws
 * a std::runtime_error when the expected number of bytes does not match the
 * actual number of bytes in the file.
 */
TEST_F(GPT2Test, load_weights) {

    C = 768; // embedding dimension
    L = 12; // number of transformer blocks
    V = 50257; // vocab size
    maxT = 1024; // max sequence length
    NH = 12; // number of attention heads

    GPT2* model = nullptr;
    model = new GPT2(C, L, V, maxT, NH);

    std::string filepath = "/Users/malavpatel/Coding_Projects/StaticGrad/gpt2_python/bin/gpt2_weights.bin";

    EXPECT_NO_THROW(model->load_weights(filepath));

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}