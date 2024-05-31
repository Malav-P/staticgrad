#include <iostream>
#include <memory>
#include <gtest/gtest.h>
#include "include/gpt2.hpp"
#include <chrono>
#include <random>

using namespace std;

void fillArrayWithRandom(float* arr, int size) {
    // Seed with a real random value, if available
    std::random_device rd;

    // Initialize the random number generator with the seed
    std::mt19937 gen(rd());

    // Define the uniform real distribution in the range [-1, 1]
    std::uniform_real_distribution<float> distrib(-1.0f, 1.0f);

    // Fill the array with random float numbers in the range [-1, 1]
    for (size_t i = 0; i < size; ++i) {
        arr[i] = distrib(gen);
    }
}


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

    EXPECT_NO_THROW(GPT2* model = new GPT2(C, L, V, maxT, NH));

    GPT2* model = new GPT2(C, L, V, maxT, NH);
    EXPECT_EQ(model->tblocks.size(), L); // expect L transformer blocks
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

    Node* out = new Node();
    out->act = acts + num_act - B*T*V;
    out->act_grads = grad + num_act - B*T*V;
    out->shape = {B, T, V};
    out->size = B*T*V;

    // Define the range
    size_t lower_bound = 0;
    size_t upper_bound = V-1;

    // Seed with a real random value, if available
    std::random_device rd;

    // Initialize random number generator with seed
    std::mt19937 gen(rd());

    // Define the distribution range
    std::uniform_int_distribution<> distrib(lower_bound, upper_bound);

    // fill input with random tokens
    for (int i =0; i < B*T; i++){
        in->act[i] = distrib(gen);
    }

    // fill parameters with random values between -1.0 and 1.0
    fillArrayWithRandom(model->params, model->num_params);

    // check outputs are zero
    for (int i = 0; i < out->size; i++){
        EXPECT_FLOAT_EQ(out->act[i], 0.0f);
    }

    // forward pass should work
    EXPECT_NO_THROW(model->forward(out, in));

    // each array at (b, t) position should sum to one
    for (int b = 0; b < B; b++){
        for (int t = 0; t < T; t++){
            float* probs = out->act + b * T*V + t * V;
            float sum = 0.0f;
            for (int v = 0; v < V; v++){
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

    Node* out = new Node();
    out->act = acts + num_act - B*T*V;
    out->act_grads = grad + num_act - B*T*V;
    out->shape = {B, T, V};
    out->size = B*T*V;

    // Define the range
    size_t lower_bound = 0;
    size_t upper_bound = V-1;

    // Seed with a real random value, if available
    std::random_device rd;

    // Initialize random number generator with seed
    std::mt19937 gen(rd());

    // Define the distribution range
    std::uniform_int_distribution<> distrib(lower_bound, upper_bound);

    // fill input with random tokens
    for (int i =0; i < B*T; i++){
        in->act[i] = distrib(gen);
    }

    std::cout << "Completed act fill\n";


    // fill parameters with random values between -1.0 and 1.0
    fillArrayWithRandom(model->params, model->num_params);

    // check outputs are zero
    for (int i = 0; i < out->size; i++){
        EXPECT_FLOAT_EQ(out->act[i], 0.0f);
    }

    std::cout << "Beginnning forward pass\n";

    // forward pass should work
    model->forward(out, in);

    std::cout << "Completed forward pass\n";

    // fill output grad
    for (int j = 0; j < out->size; j++){
        out->act_grads[j] = 1.0f;
    }

    std::cout << "Completed output grad fill\n";

    // backward pass should work
    EXPECT_NO_THROW(model->backward(out, in));

    std::cout << "Completed backward pass\n";


}

TEST_F(GPT2Test, InvalidNumHeads) {
    C = 768; // embedding dimension
    L = 12; // number of transformer blocks
    V = 50257; // vocab size
    maxT = 1024; // max sequence length
    NH = 11; // number of attention heads

    B = 4; // batch size
    T = 64; // sequence length

    EXPECT_THROW(GPT2* model = new GPT2(C, L, V, maxT, NH), std::invalid_argument);

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
    for (int i = 0; i < gpt2->num_params; i++) {
        gpt2->grad[i] = 1.0f;
    }

    // Measure the time taken by zero_grad()
    auto start = std::chrono::high_resolution_clock::now();
    gpt2->zero_grad();
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time
    std::chrono::duration<double> elapsed = end - start;

    // Print the elapsed time
    std::cout << "Time taken by zero_grad(): " << elapsed.count() << " seconds" << std::endl;


    // Check if all gradients are zero
    for (int i = 0; i < gpt2->num_params; i++) {
        EXPECT_FLOAT_EQ(gpt2->grad[i], 0.0f);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}