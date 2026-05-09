import os
import numpy as np
from tqdm.asyncio import tqdm
from cs336_basics.tokenizer import Tokenizer
from tqdm import tqdm

def encode_texts(tokenizer, input_path, output_path):
    '''
    Encode the input text file into a sequence of token IDs using the provided tokenizer and save the result to output_path.
    '''
    # if os.path.exists(output_path):
    #     print(f"Encoded data already exists: {output_path}")
    #     return output_path
    
    token_ids = []
    with open(input_path, 'r', encoding='utf-8') as f:
        with tqdm(total=os.path.getsize(input_path), unit='B', unit_scale=True, desc="Encoding texts") as pbar:
            for line in f:
                line_ids = tokenizer.encode(line)
                token_ids.extend(line_ids)
                pbar.update(len(line.encode('utf-8')))

    np.save(output_path, np.array(token_ids, dtype=np.uint16))
    print(f"Encoded data saved to {output_path}")
    return output_path

def main():
    from pathlib import Path
    import pickle
    
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
    
    # Encode the input text file and save the result
    input_path_tinystories = "./tests/fixtures/tinystories_sample_5M.txt"
    output_path_tinystories = "./output/train_ids_tinystories.npy"
    
    input_path_owt = "./data/openwebtext_sample_10lines.txt"
    output_path_owt = "./output/train_ids_owt.npy"
    
    encode_texts(tokenizer_tinystories, input_path_tinystories, output_path_tinystories)
    encode_texts(tokenizer_owt, input_path_owt, output_path_owt)
    
if __name__ == "__main__":
    main()