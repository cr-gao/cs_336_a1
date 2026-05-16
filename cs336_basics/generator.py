import torch
import argparse
import yaml
import pickle

from types import SimpleNamespace
from tokenizer import Tokenizer
from model import TransformerLM
from optimizer import AdamW
from train import load_config, dict_to_namespace, load_checkpoint

def softmax(logits, temperature=1.0, dimension=-1):
    scaled_logits = logits / temperature
    exp_logits = torch.exp(scaled_logits - torch.max(scaled_logits, dim=dimension, keepdim=True).values)
    return exp_logits / torch.sum(exp_logits, dim=dimension, keepdim=True)

def truncate_kv_cache(kv_cache, max_length):
    truncated_kv_cache = []
    for K, V in kv_cache:
        K = K[:, -max_length:, :]
        V = V[:, -max_length:, :]
        truncated_kv_cache.append((K, V))
    return truncated_kv_cache

def generate(
    model, 
    tokenizer, 
    prompt,
    context_size,
    max_new_tokens, 
    temperature=1.0, 
    top_p=1.0,
    device='cpu'
):
    model.eval()
    
    tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    
    # prefill
    logits, kv_cache = model(tokens, use_cache=True)
    for _ in range(max_new_tokens):        
        probs = softmax(logits, temperature=temperature, dimension=-1)
        sorted_probs, sorted_indices = torch.sort(probs[:, -1, :], descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        cutoff = cumulative_probs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_indices_to_keep = sorted_indices[~cutoff]
        next_token = torch.multinomial(probs[:, -1, :][:, sorted_indices_to_keep], num_samples=1)
        next_token = sorted_indices_to_keep[next_token]

        tokens = torch.cat([tokens, next_token], dim=-1)
        
        if tokenizer.decode([next_token.item()]) in tokenizer.special_tokens:
            break
        
        if kv_cache[0][0].shape[-2] > context_size:
            kv_cache = truncate_kv_cache(kv_cache, context_size)
        
        with torch.no_grad():
            logits, kv_cache = model(next_token, kv_cache=kv_cache, use_cache=True)
        
    return tokenizer.decode(tokens[0].tolist())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    
    args = parser.parse_args()
    config = load_config(args.config)
    config = dict_to_namespace(config)
    
    device = config.system.device
    model = TransformerLM(d_model=config.model.d_model, 
                          num_layers=config.model.num_layers, 
                          num_heads=config.model.num_heads, 
                          d_ff=config.model.d_ff, 
                          use_rope=config.model.use_rope, 
                          vocab_size=config.model.vocab_size, 
                          context_size=config.model.context_size, 
                          device=device).to(device)
    optimizer = AdamW(model.parameters(), 
                      lr=float(config.optimizer.learning_rate), 
                      betas=(config.optimizer.beta1, config.optimizer.beta2), 
                      eps=float(config.optimizer.eps), 
                      weight_decay=config.optimizer.weight_decay)
    load_checkpoint(args.checkpoint, model, optimizer)
    
    special_tokens = config.tokenizer.special_tokens
    vocab_file = config.tokenizer.vocab_file
    merge_file = config.tokenizer.merge_file
    with open(vocab_file, "rb") as f:
            vocab = pickle.load(f)
    with open(merge_file, "rb") as f:
        merges = pickle.load(f)
    
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    generated_text = generate(
        model, 
        tokenizer, 
        args.prompt, 
        context_size=config.model.context_size,
        max_new_tokens=config.generation.max_new_tokens, 
        temperature=config.generation.temperature, 
        top_p=config.generation.top_p,
        device=device
    )
    print(generated_text)
    
if __name__ == "__main__":
    main()
    
        
        