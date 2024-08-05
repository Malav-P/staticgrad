#include <gtest/gtest.h>
#include "classes.hpp"
#include "test_common.hpp"


class MatmulTest : public ::testing::Test {
  protected:

  void initialize(size_t B, size_t T, size_t C, size_t OC){
      in = new Node();
      in->act = new float[B * T * C];
      in->act_grads = new float[B * T * C]; 
      in->shape = {B, T, C};
      in->size = B * T * C;

      out = new Node();
      out->act = new float[B * T * OC];
      out->act_grads = new float[B * T * OC]; out->shape = {B, T, OC};
      out->size = B * T * OC;

      params = new float[OC*C];
      grad = new float[OC*C];

      fillArrayWithRandom(params, C * OC);


  }

  void SetUp() override {}
  void TearDown() override {}

  void teardown() {
        delete[] in->act;    
        delete[] in->act_grads;
        delete in;        
        delete[] out->act;        
        delete[] out->act_grads;
        delete out;
        delete[] params;
        delete[] grad;
  } 

  Node* in; 
  Node* out;  
  float* params;
  float* grad;
};

TEST_F(MatmulTest, Forward) {
   size_t B = 2;
   size_t T = 3;
   size_t C = 4;
   size_t OC = 5;

   initialize(B, T, C, OC);

   Matmul* matmul = new Matmul(params, nullptr);

   // Fill input and output arrays with random values
   fillArrayWithRandom(in->act, in->size);
   fillArrayWithRandom(out->act, out->size);

   matmul->forward(out, in);

   // Calculate the expected output using cblas_sgemm
   float* expected_out = new float[B * T * OC];
   for (size_t b = 0; b < B; b++){
        float* A = in->act + b * T * C;
        float* out_ = expected_out + b * T * OC;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, OC, C, 1.0f, A, C, params, OC, 0.0f, out_, OC);
   }

   for (size_t i = 0; i < out->size; i++) {
        EXPECT_NEAR(out->act[i], expected_out[i], 1e-5);
   }

   matmul->forward3(out, in);

   for (size_t i = 0; i < out->size; i++) {
        EXPECT_NEAR(out->act[i], expected_out[i], 1e-5);
   }

   delete[] expected_out;
   delete matmul;

   teardown();
}  

TEST_F(MatmulTest, Forward2) {
   size_t B = 1;
   size_t T = 2;
   size_t C = 3;
   size_t OC = 4;

   initialize(B, T, C, OC);

   Matmul* matmul = new Matmul(params, grad);


   // input is (1, 2, 3) tensor. [1, 2, 3
   //                             4, 5, 6]
   for (size_t i = 0; i < B*T*C; i++){
      in->act[i] = i+1;
   }

   // params is (4, 3) matrix. [7, 8, 9
   //                           10, 11, 12
   //                           13, 14, 15
   //                           16, 17, 18]
   for (size_t i = 0; i < OC*C; i++){
      params[i] = 7+i;
   }

   
   // expected it (2, 4) matrix. [50, 68, 86, 104
   //                             122, 167, 212, 257]
   float expected[8] = {50.0f, 68.0f, 86.0f, 104.0f, 122.0f, 167.0f, 212.0f, 257.0f};

   matmul->forward2(out, in);

   for (size_t i = 0; i < T*OC; i++){
      EXPECT_FLOAT_EQ(out->act[i], expected[i]);
   }

   delete matmul;
   teardown();

} 

TEST_F(MatmulTest, Backward) {
 
   size_t B = 2;
   size_t T = 3;
   size_t C = 4;
   size_t OC = 5;

   initialize(B, T, C, OC);

	Matmul* matmul = new Matmul(params, grad);

	// Fill data
	std::memset(in->act_grads, 0, in->size * sizeof(float));
	std::memset(matmul->grad, 0, C*OC * sizeof(float));
	fillArrayWithOnes(out->act_grads, out->size);
	fillArrayWithRandom(in->act, in->size);

	matmul->backward(out, in);

	// check gradient wrt input
	for (size_t b = 0; b < B; b++){
		for (size_t t = 0;  t < T; t++){
			for (size_t c = 0; c < C; c++){

				float expected = 0.0f;
				for (size_t i = 0; i < OC; i++){
					expected += matmul->params[ c*OC + i];
				}
				EXPECT_FLOAT_EQ(in->act_grads[b*T*C + t*C + c], expected);

			}

		}
	}

	// check gradient wrt param
	for (size_t i = 0; i < C; i++){

		float expected = 0.0f;
		for (size_t b = 0; b < B; b++){
			float* A = in->act + b*T*C + i;
			
			for (size_t t = 0; t < T; t++){
				expected += A[t*C];
			}
			
		}

		for (size_t j = 0 ; j < OC; j++){
			EXPECT_FLOAT_EQ(matmul->grad[i*OC + j], expected);
		}

	}


	delete matmul;
   teardown();
}

TEST_F(MatmulTest, Backward2) {
   size_t B = 2;
   size_t T = 3;
   size_t C = 4;
   size_t OC = 5;

   initialize(B, T, C, OC);

	Matmul* matmul = new Matmul(params, grad);

	// Fill data
	std::memset(in->act_grads, 0, in->size * sizeof(float));
	std::memset(matmul->grad, 0, C*OC * sizeof(float));
	fillArrayWithOnes(out->act_grads, out->size);
	fillArrayWithRandom(in->act, in->size);

	matmul->backward2(out, in);

	// check gradient wrt input
	for (size_t b = 0; b < B; b++){
		for (size_t t = 0;  t < T; t++){
			for (size_t c = 0; c < C; c++){

				float expected = 0.0f;
				for (size_t i = 0; i < OC; i++){
					expected += matmul->params[ c + i*C];
				}
				EXPECT_FLOAT_EQ(in->act_grads[b*T*C + t*C + c], expected);

			}

		}
	}

	// check gradient wrt param
	for (size_t i = 0; i < C; i++){

		float expected = 0.0f;
		for (size_t b = 0; b < B; b++){
			float* A = in->act + b*T*C + i;
			
			for (size_t t = 0; t < T; t++){
				expected += A[t*C];
			}
			
		}

		for (size_t j = 0 ; j < OC; j++){
			EXPECT_FLOAT_EQ(matmul->grad[j*C + i], expected);
		}

	}


	delete matmul;
   teardown();
}

TEST_F(MatmulTest, InputNull) {
   size_t B = 2;
   size_t T = 3;
   size_t C = 4;
   size_t OC = 5;

   initialize(B, T, C, OC);
   Matmul* matmul = new Matmul(params, nullptr);

   delete[] in->act;
   in->act = nullptr;
   EXPECT_THROW(matmul->forward(out, in), std::invalid_argument);

   delete matmul;
   teardown();
} 

TEST_F(MatmulTest, OutputNull) {
   size_t B = 2;
   size_t T = 3;
   size_t C = 4;
   size_t OC = 5;

   initialize(B, T, C, OC);
   Matmul* matmul = new Matmul(params, nullptr);

   delete[] out->act;
   out->act = nullptr;
   EXPECT_THROW(matmul->forward(out, in), std::invalid_argument);

   delete matmul;
   teardown();
}

int main(int argc, char **argv) {
       ::testing::InitGoogleTest(&argc, argv);
       return RUN_ALL_TESTS();
 }