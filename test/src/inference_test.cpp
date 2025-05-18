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
    Optimizer* opt = nullptr;
    DataStream* ds = nullptr;
    Tokenizer* tk = nullptr;
    Activation* activations = nullptr;
    Node* out = nullptr;
    Node* in = nullptr;
    size_t B = 1;
    size_t T = seqlen;
    bool pretrained = true;

    setup_model(model, pretrained);
    setup_tokenizer(tk);
    setup_activations(activations, out, in, B, T, model);

    yap(model, tk, out, in, start);

    // deallocate memory
    tear_down(model, opt, ds, tk, activations, out, in);
}