import tiktoken
import os

# Load the GPT-2 model's encoder
encoder = tiktoken.get_encoding("gpt2")
eot_token = encoder._special_tokens['<|endoftext|>']

def write_gpt2_vocab():

    parent_dir = os.path.dirname(os.path.dirname(__file__))
    output_path = os.path.join(parent_dir, "bin/gpt2_vocab.bin")

    # Check if the output path is a valid directory
    if not os.path.isdir(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok = True)
        # raise ValueError(f"Invalid output path: {output_path}")


    n = encoder.max_token_value

    # Open the output file in binary write mode
    with open(output_path, "wb") as f:
        # Iterate over the vocabulary
        for token_id in range(n):            
            # Get the string corresponding to the token ID
            tokenstr_bytes = encoder.decode_bytes([token_id])

            assert len(tokenstr_bytes) < 255

            # Write number of bytes of string as a single byte integer. (delimiters dont work, they are part of the vocab!)
            f.write(len(tokenstr_bytes).to_bytes(length=1, byteorder="big"))
            # Encode the string as bytes and write it to the file
            f.write(tokenstr_bytes)


        eot_tokenstr_bytes = encoder.decode_bytes([eot_token])
        f.write(len(eot_tokenstr_bytes).to_bytes(length=1, byteorder="big"))
        f.write(eot_tokenstr_bytes)    

    print(f"Vocabulary of size {n + 1} written to gpt2_vocab.bin")
    print(f"First token: {encoder.decode([0])}")
    print(f"Second token: {encoder.decode([1])}")
    print(f"Last token: {encoder.decode([n])}")
            


if __name__ == "__main__":
    write_gpt2_vocab()

