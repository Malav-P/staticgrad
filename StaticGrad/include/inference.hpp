#ifndef INFERENCE_HPP
#define INFERENCE_HPP

class GPT2;
class Tokenizer;
class Node;



u_int16_t next_token(GPT2*& model,
                     Tokenizer*& tk,
                     Node*& out,
                     Node*& in,
                     size_t t);
#endif // INFERENCE_HPP