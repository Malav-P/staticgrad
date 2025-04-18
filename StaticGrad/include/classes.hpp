#ifndef CLASSES_HPP
#define CLASSES_HPP

#include "node.hpp"


void shift(Node* out, Node* in, const std::vector<size_t> shape_);
void shift_back(Node* out, Node* in, const std::vector<size_t> shape_);

class Operation {
    public :
        Operation(float* params_, float* grad_):
            params(params_),
            grad(grad_) {}

        float* params;
        float* grad;

        virtual ~Operation() {}

        virtual void forward(Node* out, Node* in) = 0;
        virtual void backward(Node* out, Node* in) = 0;

};

class Embedding: public Operation {
    public:
        size_t vocab_size;
        size_t C;

        Embedding(float* params_, float* grad_, size_t C_, size_t vocab_size_):
            Operation(params_, grad_),
            vocab_size(vocab_size_),
            C(C_){}

        void forward(Node* out, Node* in) override;
        void backward(Node* out, Node* in) override;
};

class GELU : public Operation {
    public :

        GELU():
            Operation(nullptr, nullptr){}

        void forward(Node* out, Node* in) override;
        void backward(Node* out, Node* in) override;
};

class LayerNorm : public Operation {
    public:

        float* rstd;
        float* m;
        size_t size;
    

        LayerNorm(float* params_, float* grad_):
        Operation(params_, grad_),
        rstd(nullptr),
        m(nullptr),
        size(0){}

        void forward(Node* out, Node* in) override;
        void backward(Node* out, Node* in) override;


        ~LayerNorm(){
            delete[] rstd;
            delete[] m;
        }
};

class Matmul : public Operation {
    public :
        float multiplier;
        bool bias;
        
        Matmul(float* params_, float* grad_, float multiplier_ = 1.0, bool bias_ = true):
            Operation(params_, grad_),
            multiplier(multiplier_),
            bias(bias_){}
        
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

        Add():
            Operation(nullptr, nullptr){}
        
        void forward(Node* out, Node* in) override;
        void backward(Node* out, Node* in) override;

};

class RowAdd : public Operation {
    public :

        RowAdd(float* params_, float* grad_):
            Operation(params_, grad_) { }

        void forward(Node* out, Node* in) override;
        void backward(Node* out, Node* in) override;
};




class TransformerBlock : public Operation {
    public:

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

        TransformerBlock(float* params_, float* grad_, const size_t C, const size_t NH); // constructor
        ~TransformerBlock(); // destructor

        
        void forward(Node* out, Node* in) override;
        void backward(Node* out, Node* in) override;
};

#endif // CLASSES_HPP