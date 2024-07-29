#include "utils.hpp"
#include <random>
#include <fstream>



/**
 * Computes the cross-entropy loss for a given set of input probabilities and target labels.
 *
 * Args:
 *   @param out: The output node where the computed loss will be stored, a 2D tensor of shape (B, T).
 *   @param in: The input node containing the probabilities, a 3D tensor of shape (B, T, V).
 *   @param targets: An array of shape (B, T) containing the true tokenID for each sample.
 *
 * Returns:
 *   None. The computed loss is stored in the `act` array of the `out` node.
 */
void crossentropy_forward(Node* out, Node* in, u_int16_t* targets){ // (B, T, V) -> (B, T)

    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t V = in->shape[2];


    for (size_t b = 0; b < B; b++){
        for (size_t t = 0; t < T; t++){
            float* probabilities = in->act + b*T*V + t*V;
            u_int16_t target = targets[b*T + t];

            out->act[b*T + t] = -std::logf(probabilities[target]);

        }
    }
}

/**
 * Computes the gradients of the cross-entropy loss with respect to the input
 * to the softmax function.
 *
 * Args:
 *   @param out: The output of the softmax function, a 3D tensor of shape (B, T, V).
 *   @param in: The input to the softmax function, a 3D tensor of shape (B, T, V).
 *   @param targets: An array of shape (B, T) containing the true tokenID for each sample.
 *   @param temperature: the temperature parameter
 *
 * Returns:
 *   None. The gradients are accumulated in the `act_grads` array of the `in` node.
 */
void crossentropy_softmax_backward(Node* out, Node* in, u_int16_t* targets, float temperature){
    size_t B = in->shape[0];
    size_t T = in->shape[1];
    size_t V = in->shape[2];

    for (size_t b = 0; b < B; b++){
        for (size_t t = 0; t < T; t++){
            float* probabilities = out->act + b*T*V + t*V;
            u_int16_t target = targets[b*T + t];

            for (size_t v = 0; v < V; v++){
                float indicator = (v == target) ? 1.0f : 0.0f;
                float prob = probabilities[v];

                float grad = (prob - indicator) / temperature;

                in->act_grads[b*T*V + t*V + v] += grad / (B*T); // B*T normalizer is present because final loss is mean over a (B, T) array of losses that result from the cross_entropy function
            }

        }
    }
}

/**
 * Samples a token from a given probability distribution.
 *
 * Args:
 *   @param probabilities: A pointer to the probability distribution, an array of length `length`.
 *   @param length: The number of elements in the probability distribution.
 *   @param random: A boolean flag indicating whether to sample randomly or deterministically.
 *
 * Returns:
 *   The index of the sampled token.
 *
 * Throws:
 *   std::invalid_argument if the probabilities are not valid (i.e., non-negative and summing up to 1).
 */
u_int16_t sample_token(float* probabilities, size_t length, bool random){

    // Check if probabilities are valid
    float sum = 0.0;
    for (int i = 0; i < length; i++) {
        if (probabilities[i] < 0.0) {
            throw std::invalid_argument("Probabilities must be non-negative");
        }
        sum += probabilities[i];
    }

    if (std::abs(sum - 1.0) > 1e-6) {
        throw std::invalid_argument("Probabilities must sum up to 1");
    }

    int token;

    if (random){
        static std::random_device rd;
        static std::mt19937 gen(rd());

        std::discrete_distribution<> d(probabilities, probabilities + length);
        token = d(gen);
    }

    else {
        token = std::distance(probabilities, std::max_element(probabilities, probabilities + length));

    }

    return token;
    
}

/**
 * Loads weights from a file into a given buffer.
 *
 * Args:
 *   @param dest: A pointer to the buffer where the weights will be stored.
 *   @param fname: The name of the file containing the weights.
 *   @param expected_bytes: The expected number of bytes in the file, or -1 if unknown.
 *
 * Returns:
 *   None.
 *
 * Throws:
 *   std::runtime_error if the file cannot be opened, the number of bytes in the file does not match the expected number, or an error occurs while reading the file.
 */
void load_weights(float* dest, const std::string& fname, int expected_bytes){

    // Open the file in binary mode
    std::ifstream inputFile(fname, std::ios::binary);

    if (!inputFile.is_open()){
        throw std::runtime_error("could not open file");
    }

    // Get the size of the file
    inputFile.seekg(0, std::ios::end);
    std::streamsize fileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    if (expected_bytes != -1){
        if (expected_bytes != fileSize){
            throw std::runtime_error("number of bytes in file does not match expected");
        }
    }

    // Read the contents of the file into the buffer
    if (!inputFile.read(reinterpret_cast<char*>(dest), fileSize)) {
        throw std::runtime_error("error loading contents into buffer");
    }
}