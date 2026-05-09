import os
import numpy as np
from datasets import load_dataset
from cs336_basics.tokenizer import Tokenizer
from pathlib import Path
import pickle

def sample(input_path, num_lines):
    sampled_lines = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for _ in range(num_lines):
            line = f.readline()
            if not line:
                break
            sampled_lines.append(line.strip())
    
    return sampled_lines

def prepare_sample(input_path, output_path, num_lines):
    # if os.path.exists(output_path):
    #     print(f"Data already exists: {output_path}")
    #     return output_path

    print("Preparing the sample dataset...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Writing the first {num_lines} lines to disk...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in sample(input_path, num_lines):
            f.write(line + "\n")
    
    print(f"Sample dataset ready: {output_path}")
    return output_path

def encode_and_calc_compression_ratio(input_path, tokenizer):
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
        bytes_len = len(text.encode('utf-8'))
        token_ids = tokenizer.encode(text)
        tokens_len = len(token_ids)
        compression_ratio = bytes_len / tokens_len if tokens_len > 0 else float('inf')
    return compression_ratio

def main():
    input_path_tinystories = "./data/tinystories_full.txt"
    input_path_owt = "./data/openwebtext_sample.txt"
    
    output_path_tinystories = "./data/tinystories_sample_10lines.txt"
    output_path_owt = "./data/openwebtext_sample_10lines.txt"
    
    prepare_sample(input_path_tinystories, output_path_tinystories, num_lines=10)
    prepare_sample(input_path_owt, output_path_owt, num_lines=10)
    
    # Load the vocab and merges from the BPE training output
    output_dir = Path("./output")

    with open(output_dir / "vocab_tinystories.pkl", "rb") as f:
        vocab_tinystories = pickle.load(f)
    with open(output_dir / "merges_tinystories.pkl", "rb") as f:
        merges_tinystories = pickle.load(f)
    
    with open(output_dir / "vocab_owt.pkl", "rb") as f:
        vocab_owt = pickle.load(f)
    with open(output_dir / "merges_owt.pkl", "rb") as f:
        merges_owt = pickle.load(f)
    
    # Initialize the tokenizer
    tokenizer_tinystories = Tokenizer(vocab_tinystories, merges_tinystories)
    tokenizer_owt = Tokenizer(vocab_owt, merges_owt)
    
    print("Calculating compression ratio for TinyStories sample:")
    compression_ratio_tinystories = encode_and_calc_compression_ratio(output_path_tinystories, tokenizer_tinystories)
    compression_ratio_owt = encode_and_calc_compression_ratio(output_path_owt, tokenizer_owt)
    compression_ratio_tinystories_to_owt = encode_and_calc_compression_ratio(output_path_tinystories, tokenizer_owt)
    compression_ratio_owt_to_tinystories = encode_and_calc_compression_ratio(output_path_owt, tokenizer_tinystories)    
    print(f"Compression ratio for TinyStories sample: {compression_ratio_tinystories:.2f}")
    print(f"Compression ratio for OWT sample: {compression_ratio_owt:.2f}")
    print(f"Compression ratio for TinyStories to OWT: {compression_ratio_tinystories_to_owt:.2f}")
    print(f"Compression ratio for OWT to TinyStories: {compression_ratio_owt_to_tinystories:.2f}")
    
if __name__ == "__main__":
    main()