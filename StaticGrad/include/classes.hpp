#ifndef CLASSES_HPP
#define CLASSES_HPP

#include <vector>
#include <stdexcept>
#include <iostream>
#include <Accelerate/Accelerate.h>

class Node {
    public:

        float* act;
        float* act_grads;

        std::vector<size_t> shape;
        size_t size; // number of elements


        Node():
            act(nullptr),
            act_grads(nullptr),
            shape({0}){}

        ~Node(){}
};

void shift(Node* out, Node* in, std::vector<size_t> shape_);
void shift_back(Node* out, Node* in, std::vector<size_t> shape_);

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

class Encoder: public Operation {
    public:
        size_t vocab_size;
        size_t C;

        Encoder(float* params_, float* grad_, size_t C_, size_t vocab_size_):
            vocab_size(vocab_size_),
            C(C_),
            Operation(params_, grad_) {}

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
    

        LayerNorm(float* params_, float* grad_):
        rstd(nullptr),
        m(nullptr),
        Operation(params_, grad_) {}

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
        
        Matmul(float* params_, float* grad_, float multiplier_ = 1.0):
            multiplier(multiplier_),
            Operation(params_, grad_) {}
        
        void forward(Node* out, Node* in) override;
        void backward(Node* out, Node* in) override;

};

class Attention : public Operation {
    public:
        size_t num_heads;
        float* buffer;
        float* dbuffer;

        Attention(size_t num_heads_, size_t maxT):
            Operation(nullptr, nullptr),
            num_heads(num_heads_),
            buffer(nullptr),
            dbuffer(nullptr) {}

        ~Attention(){
            delete[] buffer;
            delete[] dbuffer;
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

class Softmax : public Operation {
    public:

        float temperature;

        Softmax(float temp):
            temperature(temp),
            Operation(nullptr, nullptr){}

        void forward(Node* out, Node* in) override;
        void backward(Node* out, Node* in) override;

        void set_temperature(float temp);

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

        LayerNorm* ln1;
        Matmul* mat1;
        RowAdd* ra1;
        Attention* att;
        Matmul* mat2;
        RowAdd* ra2;
        Add* res1;
        LayerNorm* ln2;
        Matmul* mat3;
        RowAdd* ra3;
        GELU* gelu;
        Matmul* mat4;
        RowAdd* ra4;
        Add* res2;

        TransformerBlock(float* params_, float* grad_, size_t C, size_t NH, size_t maxT):
            Operation(params_, grad_),
            res1_node(nullptr),
            res2_node(nullptr) {

                if (C % NH != 0){
                    throw std::invalid_argument("C must be divisible by NH");
                }

                res1_node = new Node();
                res2_node = new Node();

                float* layer_param = params_;
                float* layer_grad = grad_;


                ln1 = new LayerNorm(layer_param, layer_grad);
                layer_param += C + C;
                layer_grad += C + C;


                mat1 = new Matmul(layer_param, layer_grad);
                layer_param += C * 3*C;
                layer_grad +=  C * 3*C;


                ra1 = new RowAdd(layer_param, layer_grad);
                layer_param += 3*C;
                layer_grad += 3*C;

                att = new Attention(NH, maxT);

                mat2 = new Matmul(layer_param, layer_grad);
                layer_param += C*C;
                layer_grad +=  C*C;

                ra2 = new RowAdd(layer_param, layer_grad);
                layer_param += C;
                layer_grad += C;

                res1 = new Add();

                ln2 = new LayerNorm(layer_param, layer_grad);
                layer_param += C + C;
                layer_grad += C + C;

                mat3 = new Matmul(layer_param, layer_grad);
                layer_param += C * 4*C;
                layer_grad += C * 4*C;

                ra3 = new RowAdd(layer_param, layer_grad);
                layer_param += 4*C;
                layer_grad += 4*C;

                gelu = new GELU();

                mat4 = new Matmul(layer_param, layer_grad);
                layer_param += 4*C * C;
                layer_grad += 4*C * C;

                ra4 = new RowAdd(layer_param, layer_grad);
                layer_param += C;
                layer_grad += C;

                res2 = new Add();
            }

        ~TransformerBlock(){
            delete res2;
            delete ra4; delete mat4;
            delete gelu;
            delete ra3; delete mat3;
            delete ln2;
            delete res1;
            delete ra2; delete mat2;
            delete att;
            delete ra1; delete mat1;
            delete ln1;

            delete res1_node;
            delete res2_node;
        }

        
        void forward(Node* out, Node* in) override;
        void backward(Node* out, Node* in) override;
};

#endif // CLASSES_HPP