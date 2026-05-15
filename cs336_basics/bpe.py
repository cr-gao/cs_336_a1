import json

import regex as re
from tqdm import tqdm
import os
import regex as re
from tqdm import tqdm

from collections import Counter
from multiprocessing import Pool, cpu_count

PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

''' For passing the test cases '''
def train_bpe(input_path, vocab_size, special_tokens):
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
        # Split text according to special tokens
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
        special_patterns = "(" + "|".join(re.escape(token) for token in sorted_specials) + ")"
        final_chunks = []
        for chunk in re.split(special_patterns, text):
            if chunk in special_tokens:
                continue
            sub_chunks = re.findall(PAT, chunk)
            final_chunks.extend(sub_chunks)
            
        # Count frequencies
        from collections import Counter
        counter = Counter()
        for chunk in final_chunks:
            byte_tuple = tuple(bytes([b]) for b in chunk.encode('utf-8'))
            counter[byte_tuple] += 1
        
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


'''
Parallel version for actual model training.

def process_chunk(args):
    chunk_info, special_tokens, PAT = args
    file_path, start, end = chunk_info
    counter = Counter()
    sorted_specials = sorted(special_tokens, key=len, reverse=True)
    special_patterns = "(" + "|".join(re.escape(token) for token in sorted_specials) + ")"
    with open(file_path, 'r', encoding='utf-8') as f:
        f.seek(start)
        if(start != 0):
            f.readline()  # Skip partial line
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            for chunk in re.split(special_patterns, line):
                if chunk in special_tokens:
                    continue
                sub_chunks = re.findall(PAT, chunk)
                for sub_chunk in sub_chunks:
                    byte_tuple = tuple(bytes([b]) for b in sub_chunk.encode('utf-8'))
                    counter[byte_tuple] += 1
    return counter

def get_file_chunks(file_path, num_chunks):
    file_size = os.path.getsize(file_path)
    chunk_size = file_size // num_chunks
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else file_size
        chunks.append((file_path, start, end))
    return chunks

def get_initial_counters(file_path, special_tokens, PAT):
    num_workers = cpu_count()
    chunks = get_file_chunks(file_path, num_workers)
    
    tasks = [(chunk, special_tokens, PAT) for chunk in chunks]
    
    total_counter = Counter()
    with Pool(num_workers) as pool:
        with tqdm(total=len(chunks), desc="Multiprocessing Pre-tokenization") as pbar:
            for chunk_counter in pool.imap_unordered(process_chunk, tasks):
                total_counter.update(chunk_counter)
                pbar.update(1)

    return total_counter

def train_bpe(input_path, vocab_size, special_tokens):
    # Count frequencies -- multi-process streaming
    counter = get_initial_counters(input_path, special_tokens, PAT)
    
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
    merge_count = 0
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
        merge_count += 1
    pbar.close()
    print(f"Total merges performed: {merge_count}")
            
    return vocab, merges
'''

def save_readable_vocab_and_merges(vocab, merges, output_vocab_file, output_merge_file):
    readable_vocab = {
        str(idx): b.decode('latin-1') for idx, b in vocab.items()
    }
    readable_merges = [
        (pair[0].decode('latin-1'), pair[1].decode('latin-1')) 
        for pair in merges
    ]
    
    with open(output_vocab_file, "w", encoding="utf-8") as f:
        json.dump(readable_vocab, f, ensure_ascii=False, indent=4)
    with open(output_merge_file, "w", encoding="utf-8") as f:
        json.dump(readable_merges, f, ensure_ascii=False, indent=4)