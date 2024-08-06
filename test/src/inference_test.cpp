#include "inference.hpp"
#include "interface.hpp"
#include "gpt2.hpp"

int main(int argc, char **argv) {

    // setup model
    GPT2* model = nullptr;
    DataStream* ds = nullptr;
    Tokenizer* tk = nullptr;
    Node* out = nullptr;
    Node* in = nullptr;

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
    allocate_activations(out, in, B, T, model->C, model->L, model->V);

    yap(model, tk, out, in, start);

    deallocate_activations(out, in);
    tear_down(model, ds, tk);

}