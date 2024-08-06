import tiktoken
import requests
import numpy as np
import os
import tempfile
import tarfile
from tqdm import tqdm
import json
import concurrent.futures

# The goal of this file is to download and tokenize the tinystories dataset into a raw bytestream of u16int_t integers. The bytestream is storage efficient
# and allows faster processing

enc = tiktoken.get_encoding("gpt2")
eot_token = enc._special_tokens['<|endoftext|>']

def tokenize_tinystories():
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Define the path to save the tarball
        tarball_path = os.path.join(tmpdir, "TinyStories_all_data.tar.gz")
        
        # Download the tarball
        response = requests.get(url, stream=True)
        
        if response.status_code != 200:
            raise RuntimeError(f"GET failed with status code: {response.status_code}")
        
        with open(tarball_path, 'wb') as f:
            # Get the total file size from the headers
            total_size = int(response.headers.get('content-length', 0))
            
            # Download the file with a progress bar
            with open(tarball_path, 'wb') as f, tqdm(
                    desc="Downloading tarball",
                    total=total_size,
                    unit='iB',  # 'iB' stands for 'binary bytes' (base 1024)
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024):
                    size = f.write(chunk)
                    bar.update(size)

        print(f"Downloaded tarball to {tarball_path}")

        # Extract the contents of the tarball
        with tarfile.open(tarball_path, "r:gz") as tar:
            contentdir = os.path.join(tmpdir, "contentdir")
            tar.extractall(path=contentdir)
            print(f"Extracted tarball contents to {contentdir}")

        # List the extracted files
        extracted_files = os.listdir(contentdir)


        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for filename in extracted_files:
                file_path = os.path.join(contentdir, filename)
                futures.append(executor.submit(process_file, file_path))
            
            tokens = []
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),  desc="Tokenizing JSON files"):
                tokens.extend(future.result())


        # largest_possible_token = 2**16
        # assert all([0 <= token < largest_possible_token for token in tokens])
        print("writing to bin file...")
        tokens = np.array(tokens, dtype=np.uint16)

        parent_dir = os.path.dirname(os.path.dirname(__file__))
        output_path = os.path.join(parent_dir, "bin/tinystories.bin")
        tokens.tofile(output_path)


def process_file(file_path):
    tokens = []
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        

        for example in data:
            text = example["story"]
            text = text.strip()
            tokens.append(eot_token)
            tokens.extend(enc.encode_ordinary(text))

    return tokens

        
if __name__ == "__main__":
    tokenize_tinystories()
        




        
    


        

    
