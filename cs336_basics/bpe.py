import regex as re
import torch
from tqdm import tqdm

PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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
