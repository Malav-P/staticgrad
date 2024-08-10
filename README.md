## Installation
```bash
git clone https://github.com/Malav-P/staticgrad.git
cd staticgrad
mkdir build && cd build
cmake ..
```

## To Run Inference
```bash
python3 gpt2_python/scripts/gpt2_vocab.py  # weight vocab
python3 gpt2_python/scripts/gpt2_python.py # get weights

make inference

./bin/inference <your starting text> <seq len> # keep at ~100 tokens for reasonable inference speed
```