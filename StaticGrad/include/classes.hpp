#ifndef CLASSES_HPP
#define CLASSES_HPP

#include "node.hpp"
#include <cstdint>


typedef uint16_t fp16_t;

// precomputed gelu table for f16 (128 KB)
extern fp16_t table_gelu_f16[1 << 16];

static const float GELU_COEF_A     = 0.044715f;
static const float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

inline static float gelu_f32(float x) {
    return 0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

inline static fp16_t compute_fp32_to_fp16(float f) {
    fp16_t res;
    __fp16 tmp = f;
    memcpy(&res, &tmp, sizeof(uint16_t));
    return res;
}

inline static float compute_fp16_to_fp32(fp16_t h) {
    __fp16 tmp;
    memcpy(&tmp, &h, sizeof(uint16_t));
    return (float)tmp;
}

inline static void vec_gelu_fp32(const int c, float* x, float* y) {
    for (int i = 0; i < c; i++){
        y[i] = gelu_f32(x[i]);
    }
}

inline static void vec_gelu_fp32_use_fp16(const int c, float* x, float* y) {
    uint16_t idx;
    for (int i = 0; i < c; i++){
        if (x[i] <= -10.0f){
            y[i] = 0.0f;
        }
        else if (x[i] >= 10.0f)
        {
            y[i] = x[i];
        }
        else{
            fp16_t h = compute_fp32_to_fp16(x[i]);
            memcpy(&idx, &h, sizeof(uint16_t));
            y[i] = compute_fp16_to_fp32(table_gelu_f16[idx]);
        } 
    }
}

void init_table_gelu_f16(void);


void shift(Node* out, Node* in, const std::vector<size_t> shape_);
void shift_back(Node* out, Node* in, const std::vector<size_t> shape_);

class Operation {
    public :
        Operation(void* params_, void* grad_):
            params(params_),
            grad(grad_) {}

        void* params;
        void* grad;

        virtual ~Operation() {}

        virtual void forward(Node* out, Node* in) = 0;
        virtual void backward(Node* out, Node* in) = 0;
        virtual void set_grad(void* grad_){
            if (!grad){
                grad = grad_;
            }
            else{
                throw std::runtime_error("grad is not null, cannot set!");
            }
        }

};

class Embedding: public Operation {
    public:
        size_t vocab_size;
        size_t C;

        Embedding(void* params_, void* grad_, size_t C_, size_t vocab_size_):
            Operation(params_, grad_),
            vocab_size(vocab_size_),
            C(C_){}

        void forward(Node* out, Node* in) override;
        void backward(Node* out, Node* in) override;
};

class GELU : public Operation {
    public :
        bool inference_time_opt;

        GELU():
            Operation(nullptr, nullptr),
            inference_time_opt(false){}

        void forward(Node* out, Node* in) override;
        void backward(Node* out, Node* in) override;
};

class LayerNorm : public Operation {
    public:

        float* rstd;
        float* means;
        size_t size;
        bool inference_time_opt;
    

        LayerNorm(void* params_, void* grad_):
        Operation(params_, grad_),
        rstd(nullptr),
        means(nullptr),
        size(0),
        inference_time_opt(false){}

        void forward(Node* out, Node* in) override;
        void backward(Node* out, Node* in) override;


        ~LayerNorm(){
            delete[] rstd;
            delete[] means;
        }
};

class Matmul : public Operation {
    public :
        float multiplier;
        bool bias;
        bool inference_time_opt;
        
        Matmul(void* params_, void* grad_, float multiplier_ = 1.0, bool bias_ = true, bool inference_time_opt_ = false):
            Operation(params_, grad_),
            multiplier(multiplier_),
            bias(bias_),
            inference_time_opt(inference_time_opt_){}
        
        void forward(Node* out, Node* in) override;
        void backward(Node* out, Node* in) override;

        void forward2(Node* out, Node* in);
        void backward2(Node* out, Node* in);

};

class Attention : public Operation {
    public:
        size_t num_heads;
        float* buffer;
        float* dbuffer;

        Attention(size_t num_heads_):
            Operation(nullptr, nullptr),
            num_heads(num_heads_),
            buffer(nullptr),
            dbuffer(nullptr) {}

        ~Attention(){
            delete[] buffer;
            buffer = nullptr;

            delete[] dbuffer;
            dbuffer = nullptr;
        }

        void forward(Node* out, Node* in) override;
        void backward(Node* out, Node* in) override;
};


class Add : public Operation {
    public :
        bool inference_time_opt;
        Add():
            Operation(nullptr, nullptr),
            inference_time_opt(false){}
        
        void forward(Node* out, Node* in) override;
        void backward(Node* out, Node* in) override;

};

class RowAdd : public Operation {
    public :

        RowAdd(void* params_, void* grad_):
            Operation(params_, grad_) { }

        void forward(Node* out, Node* in) override;
        void backward(Node* out, Node* in) override;
};




class TransformerBlock : public Operation {
    public:

        size_t C; // model dimension

        Node* res1_node;
        Node* res2_node;

        Node* input;
        Node* output;

        LayerNorm* ln1;
        Matmul* mat1;
        Attention* att;
        Matmul* mat2;
        Add* res1;
        LayerNorm* ln2;
        Matmul* mat3;
        GELU* gelu;
        Matmul* mat4;
        Add* res2;

        TransformerBlock(void* params_, void* grad_, const size_t C_, const size_t NH); // constructor
        ~TransformerBlock(); // destructor

        
        void forward(Node* out, Node* in) override;
        void backward(Node* out, Node* in) override;
        void set_grad(void* grad_) override;
};

#endif // CLASSES_HPP