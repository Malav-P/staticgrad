## Installation
```bash
git clone https://github.com/Malav-P/staticgrad.git
cd staticgrad
mkdir build && cd build
cmake ..
```

## Get Weights and Vocab
```bash 
conda env create -f gpt2_python/environment.yml
conda activate staticgrad_python

python3 gpt2_python/scripts/gpt2_vocab.py  # get vocab
python3 gpt2_python/scripts/gpt2_python.py --get_weights # get weights

python3 gpt2_python/scripts/tinyshakespeare.py # download and tokenize tinyshakespeare dataset, run this if you plan to run training script

conda deactivate
```
## To Run Inference
```bash
make inference
./bin/inference <your starting text> <seq len> # keep at ~100 tokens for reasonable inference speed
```