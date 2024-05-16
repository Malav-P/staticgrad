#include <gtest/gtest.h>
#include <cmath>
#include "src/classes.hpp"

// Helper function to fill an array with random values
void fillArrayWithRandom(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = static_cast <float> (rand()) / RAND_MAX;
    }
}

class MatmulTest : public ::testing::Test {
  protected:
  void SetUp() override {
        B = 2;
        T = 3;
        C = 4;
        p = C;
        n = 5;
        multiplier = 1.0;

        in = new Node();
        in->act = new float[B * T * C];
        in->act_grads = new float[B * T * C]; 
        in->shape = {B, T, C};         in->size = B * T * C;

        out = new Node();
        out->act = new float[B * T * n];
        out->act_grads = new float[B * T * n]; out->shape = {B, T, n};
        out->size = B * T * n;          
        params = new float[p * n];

        // Initialize params array with random values
        fillArrayWithRandom(params, p * n);
  }

  void TearDown() override {
        delete[] in->act;    
        delete[] in->act_grads;
        delete in;        
        delete[] out->act;        
        delete[] out->act_grads;
        delete out;
        delete[] params;
  } 

  size_t B;
  size_t T;
  size_t C;
  size_t p;
  size_t n;
  float multiplier;
  Node* in; 
  Node* out;  
  float* params;
};

TEST_F(MatmulTest, ForwardPassMatMul) {
   Matmul* matmul = new Matmul(params, nullptr);

   // Fill input and output arrays with random values
   fillArrayWithRandom(in->act, in->size);
   fillArrayWithRandom(out->act, out->size);

   matmul->forward(out, in);

   // Calculate the expected output using cblas_sgemm
   float* expected_out = new float[B * T * n];
   for (size_t b = 0; b < B; b++){
        float* A = in->act + b * T * C;
        float* out_ = expected_out + b * T * n;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, n, C, multiplier, A, C, params, n, 0.0f, out_, n);
   }

   for (int i = 0; i < out->size; i++) {
        EXPECT_NEAR(out->act[i], expected_out[i], 1e-5);
   }

   delete[] expected_out;
   delete matmul;
}  

TEST_F(MatmulTest, InputNull) {
   Matmul* matmul = new Matmul(params, nullptr);

   delete[] in->act;
   in->act = nullptr;
   EXPECT_THROW(matmul->forward(out, in), std::invalid_argument);

   delete matmul;
} 

TEST_F(MatmulTest, OutputNull) {
   Matmul* matmul = new Matmul(params, nullptr);

   delete[] out->act;
   out->act = nullptr;
   EXPECT_THROW(matmul->forward(out, in), std::invalid_argument);

   delete matmul;
}

int main(int argc, char **argv) {
       ::testing::InitGoogleTest(&argc, argv);
       return RUN_ALL_TESTS();
 }