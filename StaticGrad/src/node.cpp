#include "node.hpp"
#include <cstring>

Node::Node():
act(nullptr),
act_grads(nullptr),
shape({0}),
size(0){}

Activation::Activation(const size_t B_,
                       const size_t T_,
                       const size_t C_,
                       const size_t L_,
                       const size_t V_):
act(nullptr),
act_grads(nullptr),
size(0),
B(B_),
T(T_),
V(V_)
{
    size = B*T;

    // encoder (B, T) -> (B, T, C)
    size += B*T*C_;

    // transformer blocks (B, T, C) -> (B, T, C)
    size += L_ * (16 * B * T * C_);

    // final layernorm (B, T, C) -> (B, T, C)
    size += B*T*C_;

    // unembedding (B, T, C) -> (B, T, V)
    size += B*T*V;

    // softmax (B, T, V) -> (B, T, V)
    size += B*T*V;

    act = new float[size];
    act_grads = new float[size];     
}

Activation::~Activation()
{
    delete[] act;
    delete[] act_grads;

    act = nullptr;
    act_grads = nullptr;
}

void Activation::zero_grad()
{
    if (act_grads != nullptr){
        std::memset(act_grads, 0, sizeof(float)*size);
    }
}

void Activation::point_nodes(Node* out, Node* in)
{
    in->act = act;
    in->act_grads = act_grads;
    in->shape = {B, T};
    in->size = B*T;

    out->act = act + size - B*T*V;
    out->act_grads = act_grads + size - B*T*V;
    out->shape = {B, T, V};
    out->size = B*T*V;
}