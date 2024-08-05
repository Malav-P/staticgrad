#ifndef NODE_HPP
#define NODE_HPP

class Node {
    public:

        float* act;
        float* act_grads;

        std::vector<size_t> shape;
        size_t size; // number of elements


        Node():
            act(nullptr),
            act_grads(nullptr),
            shape({0}),
            size(0){}

        ~Node(){}
};

#endif // NODE_HPP