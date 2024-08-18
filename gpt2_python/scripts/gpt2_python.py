import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import os
import json
import numpy as np
import warnings
import argparse


class GPT2WithIntermediateStates(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.intermediate_states = []
        self.module_name = None

    def set_hook(self, module_name):
        named_modules = dict(self.named_modules())  # Convert to dictionary for easy lookup
        if module_name in named_modules:
            self.module_name = module_name
        else:
            warnings.warn("module name not found, setting hook to None")
            self.module_name = None

    def forward(self, *args, **kwargs):
        # Reset intermediate states at the beginning of each forward pass
        self.intermediate_states = []

        def hook(module, input, output):
            self.intermediate_states.append(output)

        hooks = []

        if self.module_name is not None:
            for name, module in self.named_modules():
                if self.module_name == name:
                    hooks.append(module.register_forward_hook(hook))

        output = super().forward(*args, **kwargs)

        # Remove hooks after forward pass
        for hook in hooks:
            hook.remove()

        return output

    def get_intermediate_states(self):
        return self.intermediate_states


def get_weights(model):

    # Prepare file paths
    base_path = os.path.dirname(os.path.dirname(__file__))
    weights_file = os.path.join(base_path, f"bin/{model.config.name_or_path}_weights.bin")
    metadata_file = os.path.join(base_path, f"bin/{model.config.name_or_path}_metadata.json")

    # Open a binary file to write the weights and a JSON file for metadata
    with open(weights_file, 'wb') as f, open(metadata_file, 'w') as meta_f:
        total_params = 0
        metadata = []

        # Write each parameter's data and its metadata
        for name, param in model.named_parameters():
            param_data = param.data.cpu().numpy()
            assert(param_data.dtype == np.float32)
            total_params += param_data.size

            # Write parameter data
            f.write(param_data.tobytes(order="C"))

            # Record metadata (name, shape, and size)
            metadata.append({
                'name': name,
                'shape': param_data.shape,
                'size': param_data.size,
                'first two': (param_data.flatten()[:2]).tolist()
            })

        # Save metadata to JSON file
        json.dump(metadata, meta_f, indent=4)

    print('Total parameters written:', total_params)
    print('File size:', total_params * 4 / 1e9, 'GB')  # Assuming float32 data type


def load_weights(model, weights_file, metadata_file):
    # Load metadata
    with open(metadata_file, 'r') as meta_f:
        metadata = json.load(meta_f)

    # Load the weights
    with open(weights_file, 'rb') as f:
        for param_info in metadata:
            name = param_info['name']
            shape = tuple(param_info['shape'])
            size = param_info['size']

            # Read the parameter data
            param_data = np.frombuffer(f.read(size * 4), dtype=np.float32)
            param_data = param_data.reshape(shape)

            param_data = np.copy(param_data)

            # Assign data to the model
            param = dict(model.named_parameters())[name]
            param.data = torch.from_numpy(param_data).to(param.device)


def load_model(pretrained = True, debug=True):
    # Initialize the model and tokenizer

    if pretrained:
        if debug:
            model = GPT2WithIntermediateStates.from_pretrained('gpt2')
        else:
            model = GPT2LMHeadModel.from_pretrained('gpt2')
    else:
        config = GPT2Config(
                vocab_size=50257,        # Vocabulary size (default for GPT-2)
                n_positions=1024,        # Number of positional embeddings
                n_ctx=1024,              # Context window size
                n_embd=768,              # Dimensionality of embeddings
                num_hidden_layers=12,    # Number of transformer layers
                num_attention_heads=12   # Number of attention heads
            )
        
        if debug:
            model = GPT2WithIntermediateStates(config)
        else:
            model = GPT2LMHeadModel(config)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    return model, tokenizer

def yap(model, tokenizer):

    prompt = "I enjoy taking my dog out and"
    model_inputs = tokenizer(prompt, return_tensors='pt').to("cpu")

    model_inputs.pop("attention_mask", None)


    sample_output = model.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    top_k=0
    )

    generated_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)

    print("Output:\n" + 100 * '-')
    print(generated_text)


def print_modules(model):
    for name, module in model.named_modules():
        print(f"Module Name: {name}, Module Type: {module.__class__.__name__}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--get_weights', action='store_true', help='Extract and store GPT-2 weights')
    parser.add_argument('--modules', action='store_true', help='GPT2 Modules')
    parser.add_argument('--intermediate_states', action='store_true', help='Print out intermediate states')
    parser.add_argument('--yap', action='store_true', help='Autoregressive generation of text')

    args = parser.parse_args()
    
    model, tokenizer = load_model(pretrained=True, debug=True)

    if args.get_weights:
        get_weights(model)
    if args.intermediate_states:
        model.set_hook("transformer.h.0.ln_1")  # hook to the first residual layer
        prompt = "There was a hurricane"

        model_inputs = tokenizer(prompt, return_tensors='pt').to("cpu")
        _ = model(**model_inputs)
        intermediate_states = model.get_intermediate_states()
        print(intermediate_states[0][0, 0, :4])  
    if args.modules:
        print_modules(model)
    if args.yap:
        yap(model, tokenizer)
        

if __name__ == "__main__":
    main()


