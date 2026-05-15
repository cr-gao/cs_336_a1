from collections.abc import Iterable, Iterator
import os
import regex as re
import numpy as np
from tqdm.asyncio import tqdm

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        '''
        Construct a tokenizer from a given 
        vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept 
        the following parameters:
        vocab: dict[int, bytes]  
        merges: list[tuple[bytes, bytes]]  
        special_tokens: list[str] | None = None
        '''
        self.vocab = vocab
        self.merges = merges
        self.pat = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        existing_vocab_inv = {v: k for k, v in vocab.items()}
        self.special_tokens = {}
        if special_tokens:
            curr_max_id = max(self.vocab.keys()) + 1
            for token in special_tokens:
                if token.encode('utf-8') in existing_vocab_inv:
                    self.special_tokens[token] = existing_vocab_inv[token.encode('utf-8')]
                else:
                    self.special_tokens[token] = curr_max_id
                    self.vocab[curr_max_id] = token.encode('utf-8')
                    curr_max_id += 1
                
        self.vocab_inv = {v: k for k, v in vocab.items()}

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        '''
        Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the 
        same format that your BPE training code output) and (optionally) a list of special tokens. 
        This method should accept the following additional parameters:
        vocab_filepath: str  
        merges_filepath: str  
        special_tokens: list[str] | None = None
        '''
        # Vocab in json, merges in txt
        import json
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
            vocab = {int(k): v.encode('utf-8') for k, v in vocab.items()}
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    part1, part2 = line.strip().split()
                    merges.append((part1.encode('utf-8'), part2.encode('utf-8')))
        return cls(vocab, merges, special_tokens)
    
    def encode_chunk(self, chunk: str) -> list[int]:
        def merge_chunk(chunk_tokens, pair):
            new_chunk_tokens = []
            i = 0
            while i < len(chunk_tokens):
                if i < len(chunk_tokens) - 1 and chunk_tokens[i] == pair[0] and chunk_tokens[i+1] == pair[1]:
                    new_chunk_tokens.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_chunk_tokens.append(chunk_tokens[i])
                    i += 1
            return new_chunk_tokens
        
        token_ids = []
        if chunk in self.special_tokens:
            return self.special_tokens[chunk]
        else:
            chunk_tokens = [bytes([b]) for b in chunk.encode('utf-8')]
            for merge in self.merges:
                if len(chunk_tokens) <= 1:
                    break
                chunk_tokens = merge_chunk(chunk_tokens, merge)
                    
            for token in chunk_tokens:
                if token not in self.vocab_inv:
                    print(f"DEBUG: Missing Token = {token}")
                    print(f"DEBUG: Is it in vocab values? {token in self.vocab.values()}")
                    print(f"DEBUG: Token Type = {type(token)}")
                token_ids.append(self.vocab_inv[token])
        return token_ids
    
    def encode(self, text: str) -> list[int]:
        '''
        Encode an input text into a sequence of token IDs.
        '''
        # Split text according to special tokens
        sorted_specials = sorted(self.special_tokens.keys(), key=len, reverse=True)
        special_patterns = "(" + "|".join(re.escape(token) for token in sorted_specials) + ")"
        token_ids = []
        if self.special_tokens:
            for chunk in re.split(special_patterns, text):
                if chunk in self.special_tokens:
                    token_ids.append(self.special_tokens[chunk])
                    continue
                for match in re.finditer(self.pat, chunk):
                    curr_chunk = match.group(0)
                    token_ids.extend(self.encode_chunk(curr_chunk))
        else:
            for match in re.finditer(self.pat, text):
                curr_chunk = match.group(0)
                token_ids.extend(self.encode_chunk(curr_chunk))
                
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is 
        required for memory-efficient tokenization of large files that we cannot directly load into 
        memory.
        '''
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id
        
    def decode(self, ids: list[int]) -> str:
        '''
        Decode a sequence of token IDs into text.
        '''
        byte_list = []
        for token_id in ids:
            token_bytes = self.vocab[token_id]
            byte_list.append(token_bytes)
        return b''.join(byte_list).decode('utf-8', errors='replace')
    
def encode_texts(tokenizer, input_path):
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

    return np.array(token_ids, dtype=np.uint16)