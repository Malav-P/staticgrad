#include <vector>
#include <set>
#include <stdexcept>
#include <memory>
#include <iostream>
#include <Accelerate/Accelerate.h>

class Tensor  {
    public:
        Tensor(std::vector<size_t> shape_, std::shared_ptr<float[]> values_):
            values(values_),
            shape(shape_) {
                size_t numel = 1;

                for (size_t element : shape_) {
                    numel *= element;
                }

                grad = std::shared_ptr<float[]>(new float[numel]);
                size = sizeof(float) * numel;               
            }
        
        explicit Tensor(std::vector<size_t> shape_):
            values(nullptr),
            grad(nullptr),
            shape(shape_) {
                size_t numel = 1;

                for (size_t element : shape_) {
                    numel *= element;
                }

                values = std::shared_ptr<float[]>(new float[numel]);
                grad = std::shared_ptr<float[]>(new float[numel]); 
                size = sizeof(float) * numel;
            }

        ~Tensor(){}

        void reset_values(){
            memset(this->values.get(), 0.0, size);
        }

        void reset_grad(){
            memset(this->grad.get(), 0.0, size);
        }

        std::shared_ptr<float[]> values;
        std::shared_ptr<float[]> grad;
        std::vector<size_t> shape;
        size_t size; // in bytes

};



class Node {
    public:
        std::shared_ptr<Tensor> data;

        explicit Node(std::vector<size_t> shape_):
            data(nullptr)
            {
                data = std::shared_ptr<Tensor>(new Tensor(shape_));
            }

        ~Node(){}
};

class Operation {
    public :
        Operation(std::vector<Node*> in, Node* out):
            inputs(in),
            output(out) {}

        std::vector<Node*> inputs;
        Node* output;

        virtual ~Operation() {}

        virtual void forward() = 0;
        virtual void backward() = 0;

        virtual std::vector<size_t> output_shape() = 0;

};

class Matmul : public Operation {
    public :

        bool transpose_A;
        bool transpose_B;
        float multiplier;
        
        Matmul(Node* C, Node* A, Node* B, bool trans_A = false, bool trans_B = false, float multiplier_ = 1.0):
            transpose_A(trans_A),
            transpose_B(trans_B),
            multiplier(multiplier_),
            Operation(std::vector<Node*>{A,B}, C) {
                // TODO : checks to ensure matrix dimenions agree.
            }
        

        void forward() override;
        void backward() override;

        std::vector<size_t> output_shape() override;
};

class Add : public Operation {
    public :
        size_t numel;

        template<class... T>
        Add(Node* out, T... args):
            Operation({args...}, out){
                size_t tensor_size = this->inputs[0]->data->size;
                numel = size_t(tensor_size / sizeof(float));
                for (Node* node : this->inputs){
                    if (node->data->size != tensor_size){
                        throw std::logic_error("Sizes of operands not equal!");
                    }
                }
            }
        
        void forward() override;
        void backward() override;

        std::vector<size_t> output_shape() override;
};

class ColumnAdd : public Operation {
    public :
        ColumnAdd(Node* C, Node* A, Node* B):
            Operation(std::vector<Node*>{A,B}, C) {
                if (B->data->shape[0] != A->data->shape[0]){
                    throw std::logic_error("Sizes of ColumnAdd operation are incompatible");
                }
            }

        void forward() override;
        void backward() override;

        std::vector<size_t> output_shape() override;
};

class SoftMax : public Operation {
    public : 
        SoftMax(Node* B, Node* A):
            Operation(std::vector<Node*>{A}, B) {
                if (A->data->shape.size() != 2){
                    throw std::logic_error("Size of input tensor is not 2, SoftMax Operation cannot be done on this tensor.");
                }
            }

        void forward() override;
        void backward() override;

        std::vector<size_t> output_shape() override {return inputs[0]->data->shape;}
};

