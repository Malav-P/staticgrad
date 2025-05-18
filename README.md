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
- After first model forward pass, do not recompute activations in the transformer block. The following computations are not redone for previous tokens in the transformer blocks:
    - **Opt Type 1** : keys, values (KV caching) 
    - **Opt Type 2** : Linear layer activations
    - **Opt Type 3** : GELU activations
    - **Opt Type 4** : Residual connection activations

- **Opt Type 5** : Use a precomputed table of GELU activations. Specifically, precompute GELU for all possible fp16 values. Since there are 2^16 possible fp16 values, and each fp16 values takes 2 bytes of memory, we need a 2^17 = 128 KB table.

| KV Cache   | Linear Layer current token only | GELU current token only   | Residual Connection current token only   | FP16 GELU Table   | ms/token |
|-----|-----|-----|-----|-----|----------|
|     |     |     |     |     | 128.167  |
| ✅  | ✅  |     | ✅  |     | 104.335  |
| ✅  | ✅  |     | ✅  | ✅  | 79.3699  |
| ✅  | ✅  | ✅  | ✅  |     | 59.4481  |
| ✅  | ✅  | ✅  | ✅  | ✅  | 44.6622  |

