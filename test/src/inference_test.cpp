#include "inference.hpp"
#include "interface.hpp"
#include "gpt2.hpp"

int main(int argc, char **argv) {

    // setup model
    GPT2* model = nullptr;
    DataStream* ds = nullptr;
    Tokenizer* tk = nullptr;
    Node* out = new Node();
    Node* in = new Node();
    size_t B = 1;
    size_t T = 64;
    std::string start;

    if (argc == 2){
        start = std::string(argv[1]);
    }
    else {
        start = "This is the default start string";
    }

    bool pretrained = true;
    setup(model, ds, tk, pretrained);
    auto activations = new Activation(B, T, model->C, model->L, model->V);
    activations->point_Nodes(out, in);

    yap(model, tk, out, in, start);

    delete activations;
    tear_down(model, ds, tk);
    delete out;
    delete in;

}