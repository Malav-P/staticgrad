#ifndef NODE_HPP
#define NODE_HPP

#include <vector>
using namespace std;

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
        size_t num_bytes; ///< number of bytes used to hold activations.
        size_t B; // current batch size
        size_t T; // current sequence length
        size_t V; // vocab size

        Activation(const size_t B_, const size_t T_, const size_t C_, const size_t L_, const size_t V_);
        ~Activation();

        void zero_grad();
        void point_nodes(Node* in, Node* out);

        void reset();

};

#endif // NODE_HPP