#include "./classes.hpp"
#include <cmath>

        
void Matmul::forward() {

    size_t m = inputs[0]->data->shape[0],  k = inputs[0]->data->shape[1];
    size_t p = inputs[1]->data->shape[0],  n = inputs[1]->data->shape[1];
    float alpha = multiplier, beta = 0.0f;

    float* A = inputs[0]->data->values.get();
    float* B = inputs[1]->data->values.get();
    float* C = output->data->values.get();

    cblas_sgemm(CblasRowMajor, transpose_A ? CblasTrans : CblasNoTrans, transpose_B ? CblasTrans : CblasNoTrans,
                transpose_A ? k : m, transpose_B? p:n, transpose_A ? m : k, alpha, A, k, B, n, beta, C, transpose_B ? p:n);
}

void Matmul::backward() {
    // Compute C <- alpha * op(A) * op(B) + beta*C
    // where op() can be the identity operation or the transpose operation.
    size_t m = inputs[0]->data->shape[0], k = inputs[0]->data->shape[1];
    size_t p = inputs[1]->data->shape[0], n = inputs[1]->data->shape[1];
    float alpha = multiplier, beta = 0.0f;

    float* A = inputs[0]->data->values.get();
    float* B = inputs[1]->data->values.get();
    float* dLdA = inputs[0]->data->grad.get();
    float* dLdB = inputs[1]->data->grad.get();
    float* dLdC = output->data->grad.get();


    if (transpose_B && transpose_A){
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                    m, k, p, alpha, B, n, dLdC, p, beta, dLdA, k);

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                    p, n, k, alpha, dLdC, p, A, k, beta, dLdB, n);
    }
    else if(transpose_B && !transpose_A){
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, k, p, alpha, dLdC, p, B, n, beta, dLdA, k);
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    p, n, m, alpha, dLdC, p, A, k, beta, dLdB, n);
    }
    else if(!transpose_B && transpose_A){
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    m, k, n, alpha, B, n, dLdC, n, beta, dLdA, k);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    p, n, k, alpha, A, k, dLdC, n, beta, dLdB, n);
    }

    else if (!transpose_B && !transpose_A){
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    m, k, n, alpha, dLdC, n, B, n, beta, dLdA, k);
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    p, n, m, alpha, A, k, dLdC, n, beta, dLdB, n);
    }

    else{
        // THROW ERROR
    }
}

std::vector<size_t> Matmul::output_shape(){

    size_t rows = transpose_A ? this->inputs[0]->data->shape[1] : this->inputs[0]->data->shape[0];
    size_t cols = transpose_B ? this->inputs[1]->data->shape[0] : this->inputs[1]->data->shape[1];

    return {rows, cols};
}

void Add::forward(){
    output->data->reset_values();

    for (Node* input : inputs){
        cblas_saxpy(numel, 1, input->data->values.get(), 1, output->data->values.get(),1);
    }
}

void Add::backward(){

    std::set<Node*> inputs_no_duplicates(inputs.begin(), inputs.end());
    for (Node* input : inputs_no_duplicates){
        input->data->reset_grad();
    }

    for (Node* input: inputs){
        cblas_saxpy(numel, 1, output->data->grad.get(), 1, input->data->grad.get(), 1);
    }
    
}

std::vector<size_t> Add::output_shape(){
    return {this->inputs[0]->data->shape[0], this->inputs[0]->data->shape[1]};
}

void ColumnAdd::forward(){

    size_t nrows = inputs[0]->data->shape[0];
    size_t ncols = inputs[0]->data->shape[1];

    float* out = output->data->values.get();
    float* inp0 = inputs[0]->data->values.get();
    float* inp1 = inputs[1]->data->values.get();

    for (size_t i = 0; i < ncols; i++){
        for (size_t j = 0; j < nrows; j++){
            out[i + j*ncols]= inp0[i + j*ncols] + inp1[j];
        }
    }
}

void ColumnAdd::backward(){

    float* inp0_g = inputs[0]->data->grad.get();
    float* inp1_g = inputs[1]->data->grad.get();
    float* out_g = output->data->grad.get();

    // backward pass for first input
    if (out_g != inp0_g){
        memcpy(inp0_g, out_g, output->data->size);
    }


    // backward pass for second input
    size_t nrows = inputs[0]->data->shape[0];
    size_t ncols = inputs[0]->data->shape[1];

    for (size_t i = 0; i  < nrows; i++){
        for(size_t j = 0; j < ncols; j++){
            inp1_g[i] += out_g[i*ncols + j];
        }
    }
}

std::vector<size_t> ColumnAdd::output_shape(){
    return {this->inputs[0]->data->shape[0], this->inputs[1]->data->shape[1]};
}

void SoftMax::forward() {

    std::vector<size_t> shape = inputs[0]->data->shape;
    size_t nrows = shape[0];
    size_t ncols = shape[1];


    float sum;
    float* inp0 = inputs[0]->data->values.get();
    float* out = output->data->values.get();

    for (size_t j = 0; j < ncols; j++){
        sum = 0;

        for (size_t i = 0; i < nrows; i++){
            out[i*ncols + j] = std::exp(inp0[i*ncols + j]);
            sum += out[i*ncols + j];
        }
        
        for (size_t i = 0; i < nrows; i++){
            out[i*ncols + j] = out[i*ncols + j]/sum;
        }
    }
}

void SoftMax::backward() {

    size_t nrows = inputs[0]->data->shape[0];
    size_t ncols = inputs[0]->data->shape[1];

    float* inp0_g = inputs[0]->data->grad.get();
    float* out_g = output->data->grad.get();
    float* out = output->data->values.get();

    float softmax_deriv;
    

    for (size_t j = 0; j < ncols; j++) {
        for (size_t i = 0; i < nrows; i++) {
            // Compute the derivative of softmax
            for (size_t k = 0; k < nrows; k++){
                softmax_deriv = out[i * ncols + j] * (i==k ? 1.0:0.0 - out[i * ncols + j]);
                inp0_g[i * ncols + j] += softmax_deriv * out_g[k * ncols + j]; // WRONG WHEN BOTH POINTERS ARE THE SAME
            }            
        }
    }
}
