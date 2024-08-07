#include "inference.hpp"
#include "interface.hpp"

int main(int argc, char **argv) {

    // parse command line args
    std::string start;
    size_t seqlen;
    if (argc == 3){
        start = std::string(argv[1]);
        seqlen = std::stoi(argv[2]);
    }
    else {
        start = "This is the default start string";
        seqlen = 64;
    }

    // setup model pointers and data structures
    GPT2* model = nullptr;
    DataStream* ds = nullptr;
    Tokenizer* tk = nullptr;
    Activation* activations = nullptr;
    Node* out = nullptr;
    Node* in = nullptr;
    size_t B = 1;
    size_t T = seqlen;
    bool pretrained = true;

    // malloc and point memory
    setup(model, ds, tk, activations, out, in , B, T, pretrained);

    // autoregressive generation
    yap(model, tk, out, in, start);

    // deallocate memory
    tear_down(model, ds, tk, activations, out, in);
}