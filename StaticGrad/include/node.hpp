#ifndef NODE_HPP
#define NODE_HPP

#include <vector>

class Node {
    public:

        float* act;
        float* act_grads;

        std::vector<size_t> shape;
        size_t size; // number of elements

        size_t current_T;


        Node();
        ~Node(){}

};

class Activation {
    public:

        float* act;
        float* act_grads;

        size_t size; // number of elements
        size_t B; // current batch size
        size_t T; // current sequence length
        size_t V; // vocab size

        Activation(size_t B_, size_t T_, size_t C_, size_t L_, size_t V_);
        ~Activation();

        void zero_grad();
        void point_Nodes(Node* in, Node* out);

        
};

#endif // NODE_HPP