## Overview
This repository contains an implementation of the GPT2 architecture from scratch in C++. We have some auxiliary python code that does the following:
- downloads weights in order to load a pretrained model using this source code
- downloads and tokenizes datasets for training / finetuning

## Installation
```bash
git clone https://github.com/Malav-P/staticgrad.git
cd staticgrad
mkdir build && cd build
cmake ..
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