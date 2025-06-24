## Overview
This repository contains an implementation of the GPT2 architecture from scratch in C++. We have some auxiliary python code that does the following:
- downloads weights in order to load a pretrained model using this source code
- downloads and tokenizes datasets for training / finetuning

## Installation
```bash
git clone https://github.com/Malav-P/staticgrad.git
cd staticgrad
mkdir build && cd build
cmake .. -DBUILD_TESTING=OFF # set to ON to build tests
make install # optional, for downstream users who want to use StaticGrad libary
```

## Get Weights and Vocab
```bash 
cd ../
conda env create -f gpt2_python/environment.yml
conda activate staticgrad_python

python3 gpt2_python/scripts/gpt2_vocab.py  # get vocab
python3 gpt2_python/scripts/gpt2_python.py --get_weights # get weights

python3 gpt2_python/scripts/tinyshakespeare.py # download and tokenize tinyshakespeare dataset, run this if you plan to run training script

conda deactivate
```
## To Run Inference
```bash
cd build
make inference
./bin/inference <your starting text> <seq len> # keep at ~100 tokens for reasonable inference speed
```

## Notes on Speeding Up Inference
- After first model forward pass, do not recompute activations in the transformer block for previous tokens. The following computations in the transformer blocks are not redone for previous tokens : 
    - **Opt Type 1** : Query/Key dot products in attention block
    - **Opt Type 2** : Linear layer activations. Note that this excludes the unembedding layer since it is not in a transformer block.
    - **Opt Type 3** : GELU activations
    - **Opt Type 4** : Residual connection activations
    - **Opt Type 5** : Layernorm activations
    - **Opt Type 6** : Unembedding (Language Modeling Head) activations
    

- **Opt Type 7** : Use a precomputed table of GELU activations. Specifically, precompute GELU for all possible fp16 values. Since there are 2^16 possible fp16 values, and each fp16 value takes 2 bytes of memory, we need a 2^17 = 128 KB table.

The results of each optimization are tested by running `./bin/inference "hello" 200`
| Query/Key Dot Product Activations  | Linear Layer Activations | GELU Activations  | Residual Connection Activations  | Layernorm Activations | Unembedding Activations | FP16 GELU Table | ms/token |
|-----|-----|-----|-----|-----|-----|-----|----------|
|     |     |     |     |     || | 414.156 |
| ✅  |   |     |   |     | || 118.440  |
| ✅  | ✅  |     |   |   | || 97.353  |
| ✅  | ✅  | ✅  |   |     | | |47.386  |
| ✅  | ✅  | ✅  | ✅  |   | || 45.841  |
| ✅  | ✅  | ✅  | ✅  | ✅  | || 28.220  |
| ✅  | ✅  | ✅  | ✅  | ✅  |✅ || 26.324  |
| ✅  | ✅  | ✅  | ✅  | ✅  |✅| ✅ | 25.847  |

