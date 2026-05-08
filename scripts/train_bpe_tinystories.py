
import os
import pickle
import time
from pathlib import Path
import json

import regex as re
from cs336_basics.bpe import train_bpe
from datasets import load_dataset
from tqdm import tqdm

PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe_on_tinystories(input_path, vocab_size, special_tokens):
    # Split text according to special tokens
    special_patterns = "(" + "|".join(re.escape(token) for token in special_tokens) + ")"
    
    # Count frequencies
    from collections import Counter
    counter = Counter()
    with open(input_path, 'r', encoding='utf-8') as f:
        with tqdm(total=os.path.getsize(input_path), unit='B', unit_scale=True, desc="Reading data") as pbar:
            for line in f:
                for chunk in re.split(special_patterns, line):
                    if chunk in special_tokens:
                        continue
                    sub_chunks = re.findall(PAT, chunk)
                    for sub_chunk in sub_chunks:
                        byte_tuple = tuple(bytes([b]) for b in sub_chunk.encode('utf-8'))
                        counter[byte_tuple] += 1
                    
                pbar.update(len(line.encode('utf-8')))
    
    pairs = Counter()
    pair_to_words = {}
    for word_tuple, freq in counter.items():
        for i in range(len(word_tuple)-1):
            pair = (word_tuple[i], word_tuple[i+1])
            pairs[pair] += freq
            pair_to_words.setdefault(pair, set()).add(word_tuple)
    
    # base vocab
    vocab = {i: bytes([i]) for i in range(256)}
    for(i, token) in enumerate(special_tokens, start=256):
        vocab[i] = token.encode('utf-8')

    print("Base vocab initialized. Starting merge operations...\n")
    pbar = tqdm(total=vocab_size - 256 - len(special_tokens))
    merges = []
    while(len(vocab) < vocab_size):
        if not pairs:
            break
        best_pair = max(pairs, key = lambda p: (pairs[p], p))
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        
        # Merge the best pair in the pair counter
        word_list = list(pair_to_words[best_pair])
        for word_tuple in word_list:
            # Remove all the contributions from the word tuple
            for i in range(len(word_tuple)-1):
                pair = (word_tuple[i], word_tuple[i+1])
                pairs[pair] -= counter[word_tuple]
                pair_to_words[pair].discard(word_tuple)
                if pairs[pair] == 0:
                    del pairs[pair]
                    del pair_to_words[pair]
            
            # Merge the best pair in the word tuple
            new_word_tuple = []
            i = 0
            while(i < len(word_tuple)):
                if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i+1]) == best_pair:
                    new_word_tuple.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_word_tuple.append(word_tuple[i])
                    i += 1
            new_word_tuple = tuple(new_word_tuple)
            
            # Add the contributions from the new word tuple
            for i in range(len(new_word_tuple)-1):
                pair = (new_word_tuple[i], new_word_tuple[i+1])
                pairs[pair] += counter[word_tuple]
                pair_to_words.setdefault(pair, set()).add(new_word_tuple)
                
            counter[new_word_tuple] = counter[word_tuple]
            del counter[word_tuple]
        
        pbar.update(1)
    pbar.close()
            
    return vocab, merges

def prepare_full_tinystories(output_path="./data/tinystories_full.txt"):
    if os.path.exists(output_path):
        print(f"Data already exists: {output_path}")
        return output_path

    print("Downloading from Hugging Face and preparing the dataset...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load dataset from Hugging Face
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    print("Writing to disk... This might take a few minutes.")
    with open(output_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(dataset):
            f.write(example["text"] + "\n")
            if (i + 1) % 100000 == 0:
                print(f"Processed {i + 1} stories...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for text in dataset["text"]:
            f.write(text + "\n")
            
    print(f"Full dataset ready: {output_path}")
    return output_path

def save_readable_vocab_and_merges(vocab, merges, output_dir):
    readable_vocab = {
        str(idx): b.decode('latin-1') for idx, b in vocab.items()
    }
    readable_merges = [
        (pair[0].decode('latin-1'), pair[1].decode('latin-1')) 
        for pair in merges
    ]
    
    with open(output_dir / "vocab_readable.json", "w", encoding="utf-8") as f:
        json.dump(readable_vocab, f, ensure_ascii=False, indent=4)
    with open(output_dir / "merges_readable.json", "w", encoding="utf-8") as f:
        json.dump(readable_merges, f, ensure_ascii=False, indent=4)

def main():
    input_path = prepare_full_tinystories("./data/tinystories_full.txt")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)\

    start_time = time.time()
    vocab, merges = train_bpe_on_tinystories(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    print(f"Training BPE on {input_path} took {end_time - start_time:.2f} seconds.")
    print(f"Learned vocab of size {len(vocab)} and {len(merges)} merges.")
    
    # Save the vocab and merges to the output directory
    with open(output_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(output_dir / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    
    save_readable_vocab_and_merges(vocab, merges, output_dir)
        
    print(f"Saved vocab and merges to {output_dir / 'vocab.pkl'} and {output_dir / 'merges.pkl'}")

if __name__ == "__main__":
    main()