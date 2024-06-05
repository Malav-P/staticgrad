#include <gtest/gtest.h>
#include "gpt2.hpp"
#include "test_common.hpp"
#include <random>

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

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, V-1);

    for (int i = 0; i < B*T; i++) {
        in->act[i] = distrib(gen);
    }

    // fill parameters with random values between -1.0 and 1.0
    fillArrayWithRandom(model->params, model->num_params);

    // check outputs are zero
    for (int i = 0; i < out->size; i++){
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


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, V-1);

    for (int i = 0; i < B*T; i++) {
        in->act[i] = distrib(gen);
    }

    // fill parameters with random values between -1.0 and 1.0
    fillArrayWithRandom(model->params, model->num_params);

    // check outputs are zero
    for (int i = 0; i < out->size; i++){
        EXPECT_FLOAT_EQ(out->act[i], 0.0f);
    }

    // forward pass should work
    model->forward(out, in);


    // fill output grad
    for (int j = 0; j < out->size; j++){
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

    for (int i = 0; i < gpt2->num_params; i++){
        EXPECT_FLOAT_EQ(gpt2->m[i], (1 - gpt2->beta1) * gpt2->grad[i]);
        EXPECT_FLOAT_EQ(gpt2->v[i], (1 - gpt2->beta2) * gpt2->grad[i] * gpt2->grad[i] );
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}