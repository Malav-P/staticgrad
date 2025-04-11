#include "node.hpp"
#include <cstring>

/**
 * @brief Constructs a Node with default-initialized attributes.
 * 
 * Initializes activation pointers to `nullptr`, shape to `{0}`, and size-related 
 * attributes to zero.
 */
Node::Node():
act(nullptr),
act_grads(nullptr),
shape({0}),
size(0),
current_T(0){}

/**
 * @brief Allocates memory for activations and gradients in a transformer model.
 * 
 * This constructor calculates the total required buffer size for activations 
 * (`act`) and gradients (`act_grads`) based on the model's batch size, sequence 
 * length, hidden dimension, number of transformer layers, and vocabulary size. 
 * 
 * @param[in] B_ Batch size.
 * @param[in] T_ Sequence length.
 * @param[in] C_ Hidden dimension (embedding size).
 * @param[in] L_ Number of transformer blocks.
 * @param[in] V_ Vocabulary size.
 * 
 * @note The allocated memory is used for embedding, transformer blocks, layer 
 *       normalization, and unembedding.
 */
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

    act = new float[size];
    act_grads = new float[size]; 
    
    this->reset();
}

void Activation::reset(){
    // experimental: touch memory to improve cache timing
    memset(act, 0, size*sizeof(float));
}


/**
 * @brief Frees allocated memory for activations and gradients.
 * 
 * This destructor releases memory associated with `act` and `act_grads`, ensuring 
 * proper cleanup to prevent memory leaks.
 */
Activation::~Activation()
{
    delete[] act;
    delete[] act_grads;

    act = nullptr;
    act_grads = nullptr;
}

/**
 * @brief Resets the gradient buffer to zero.
 * 
 * If the activation gradient buffer (`act_grads`) is allocated, this function 
 * sets all gradient values to zero using `std::memset`.
 */
void Activation::zero_grad()
{
    if (act_grads != nullptr){
        std::memset(act_grads, 0, sizeof(float)*size);
    }
}

/**
 * @brief Assigns activation buffers to input and output nodes.
 * 
 * This function sets activation and gradient buffers for the input (`in`) and 
 * output (`out`) nodes, adjusting their shapes and sizes accordingly.
 * 
 * @param[out] out Pointer to the output node, which receives the final activations.
 * @param[out] in Pointer to the input node, which holds intermediate activations.
 * 
 * @note 
 * - The input node (`in`) is assigned the initial activation buffer.
 * - The output node (`out`) is assigned the final layer's activation buffer.
 * - Shape and size attributes are adjusted based on batch size (`B`), sequence 
 *   length (`T`), and vocabulary size (`V`).
 */
void Activation::point_nodes(Node* out, Node* in)
{
    in->act = act;
    in->act_grads = act_grads;
    in->shape = {B, T};
    in->size = B*T;
    in->current_T = T;

    out->act = act + size - B*T*V;
    out->act_grads = act_grads + size - B*T*V;
    out->shape = {B, T, V};
    out->size = B*T*V;
    out->current_T = T;
}