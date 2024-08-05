#include "inference.hpp"
#include "interface.hpp"

int main(int argc, char **argv) {

    // setup model
    GPT2* model = nullptr;
    DataStream* ds = nullptr;
    Tokenizer* tk = nullptr;
    Node* out = nullptr;
    Node* in = nullptr;

    size_t B = 1;
    size_t T = 64;

    bool pretrained = true;

    setup(model, ds, tk, out, in, B, T, pretrained);

    std::string start("I enjoy taking my dog out and");

    yap(model, tk, out, in, start);

}