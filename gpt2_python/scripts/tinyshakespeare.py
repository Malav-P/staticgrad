import tiktoken
import requests
import numpy as np
import os
import tempfile
from tqdm import tqdm
from transformers import GPT2Tokenizer
import argparse


# The goal of this file is to download and tokenize the tinyshakespeare dataset into a raw bytestream of u16int_t integers. The bytestream is storage efficient
# and allows faster processing




def tokenize_tinyshakespeare(tokenizer = "tiktoken"):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Define the path to save the tarball
        download_file = os.path.join(tmpdir, "tinyshakespeare.txt")
        
        # Download the tarball
        response = requests.get(url, stream=True)
        
        if response.status_code != 200:
            raise RuntimeError(f"GET failed with status code: {response.status_code}")
        
        text = "<|endoftext|>" + response.text
        text = text.strip()
        text = text.replace('\n\n', '\n\n<|endoftext|>')

        if tokenizer == "tiktoken":
            enc = tiktoken.get_encoding("gpt2")
            eot_token = enc._special_tokens['<|endoftext|>']
            tokens = enc.encode(text, allowed_special={'<|endoftext|>'})

        elif tokenizer == "transformers":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokens = tokenizer(text)["input_ids"]

        else: # default to tiktoken
            enc = tiktoken.get_encoding("gpt2")
            eot_token = enc._special_tokens['<|endoftext|>']
            tokens = enc.encode(text, allowed_special={'<|endoftext|>'})

        tokens = np.array(tokens, dtype=np.uint16)[32768:]

        parent_dir = os.path.dirname(os.path.dirname(__file__))
        output_path = os.path.join(parent_dir, "bin/tinyshakespeare.bin")
        tokens.tofile(output_path)
        

        
if __name__ == "__main__":

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Process command-line arguments.")
    
    # Define a positional argument that captures the first argument as a string
    parser.add_argument('first_argument', type=str, help="The first argument as a string")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Access the first argument
    tokenizer = args.first_argument

    tokenize_tinyshakespeare(tokenizer=tokenizer)
        