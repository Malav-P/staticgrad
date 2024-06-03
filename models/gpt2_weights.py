import torch
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer
import os

def get_weights():

    # Initialize the model and tokenizer
    model = GPT2Model.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    fname = os.path.join(os.path.dirname(__file__), f"{model.config.name_or_path}.bin")
    # Open a binary file to write the weights
    with open(fname, 'wb') as f:
        total_params = 0
        for name, param in model.named_parameters():
            param_data = param.data.cpu().numpy()
            assert(param_data.dtype == np.float32)
            total_params += param_data.size
            f.write(param_data.tobytes())

    print('Total parameters written:', total_params)
    print('File size:', total_params * 4 / 1e9, ' GB')  # Assuming float32 data type, which is 4 bytes per parameter


if __name__ == "__main__":
    get_weights()