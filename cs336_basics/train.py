import numpy as np
import torch
import yaml
import argparse
import pickle
import time
from tqdm import tqdm
from types import SimpleNamespace
from pathlib import Path

from bpe import train_bpe, save_readable_vocab_and_merges
from tokenizer import Tokenizer, encode_texts
from model import TransformerLM
from optimizer import AdamW, cross_entropy, learning_rate_schedule, gradient_clipping

def get_batch(x: np.array, batch_size: int, context_length: int, device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, len(x) - context_length, size=batch_size)

    input_batch = np.stack([x[start:start + context_length] for start in starts])
    target_batch = np.stack([x[start + 1:start + context_length + 1] for start in starts])
    
    input_batch = torch.from_numpy(input_batch).to(torch.long).to(device)
    target_batch = torch.from_numpy(target_batch).to(torch.long).to(device)
    
    return input_batch, target_batch

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)
    
def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(
            **{
                k: dict_to_namespace(v)
                for k, v in d.items()
            }
        )
    return d

@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, device):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        data = (train_data if split == "train" else val_data)

        for _ in range(config.logging.eval_iters):
            x, y = get_batch(
                data,
                config.training.batch_size,
                config.model.context_size,
                device
            )

            logits = model(x)
            loss = cross_entropy(logits, y)
            losses.append(loss.item())

        out[split] = np.mean(losses)

    model.train()
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    config = load_config(args.config)
    config = dict_to_namespace(config)
    set_seed(config.system.seed)
    
    dataset_path = config.data.dataset_path
    vocab_size = config.tokenizer.vocab_size
    special_tokens = config.tokenizer.special_tokens

    device = config.system.device
    vocab_file = config.tokenizer.vocab_file
    merge_file = config.tokenizer.merge_file
    
    if Path(vocab_file).exists() and Path(merge_file).exists():
        print("Loading existing vocab and merges...")
        with open(vocab_file, "rb") as f:
            vocab = pickle.load(f)
        with open(merge_file, "rb") as f:
            merges = pickle.load(f)
    else:
        # Train BPE and construct tokenizer
        print("Training bpe...\n")
        vocab, merges = train_bpe(dataset_path, vocab_size, special_tokens)
        # Save the vocab and merges to the output directory
        with open(vocab_file, "wb") as f:
            pickle.dump(vocab, f)
        with open(merge_file, "wb") as f:
            pickle.dump(merges, f)
        save_readable_vocab_and_merges(vocab, merges, config.tokenizer.readable_vocab_file, config.tokenizer.readable_merge_file)
        
    tokenizer = Tokenizer(vocab, merges, special_tokens) 
    
    if Path(config.data.train_path).exists() and Path(config.data.val_path).exists():
        print("Encoded train and val data already exist. Skipping encoding step...")
    else:
        # Load dataset and encode
        token_ids = encode_texts(tokenizer, dataset_path)
        split_idx = int(0.9 * len(token_ids))
        np.save(config.data.train_path, token_ids[:split_idx])
        np.save(config.data.val_path, token_ids[split_idx:])
        print(f"Encoded data saved to {config.data.train_path} and {config.data.val_path}")
    
    # Construct model and optimizer
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
    
    start_iter = 0
    if args.resume is not None:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from iteration {start_iter}")
    
    # Training loop
    train_data = np.load(config.data.train_path, mmap_mode='r')
    val_data = np.load(config.data.val_path, mmap_mode='r')
    model.train()
    for iteration in tqdm(range(start_iter, config.training.max_steps)):
        iter_time_start = time.time()
        input_batch, target_batch = get_batch(train_data, config.training.batch_size, config.model.context_size, device)
        
        lr = learning_rate_schedule(iteration, 
                                    float(config.scheduler.max_lr), 
                                    float(config.scheduler.min_lr), 
                                    config.scheduler.warmup_steps, 
                                    config.training.max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        
        logits = model(input_batch)
        loss = cross_entropy(logits, target_batch)    
        loss.backward()
        
        # Clip gradients
        grad_norm = gradient_clipping(model.parameters(), config.training.grad_clip)
        optimizer.step()
        
        iter_time = time.time() - iter_time_start
        tokens_per_sec = (config.training.batch_size * config.model.context_size) / iter_time
        
        if iteration > 0 and iteration % config.checkpoint.save_every == 0:
            save_checkpoint(model, optimizer, iteration, f"{config.checkpoint.save_dir}/checkpoint_latest.pt")
            
        if iteration % config.logging.log_every == 0:
            tqdm.write(
                f"iter {iteration:6d} | "
                f"loss {loss.item():.4f} | "
                f"lr {lr:.2e} | "
                f"grad {grad_norm:.2f} | "
                f"{tokens_per_sec:.0f} tok/s | "
                f"{iter_time:.3f} s/iter"
            )
            
        if iteration % config.logging.eval_every == 0:
            losses = estimate_loss(model, train_data, val_data, config, device)
            tqdm.write(
                f"iter {iteration:6d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"lr {lr:.2e} | "
                f"grad {grad_norm:.2f} | "
                f"{tokens_per_sec:.0f} tok/s | "
                f"{iter_time:.3f} s/iter"
            )

    save_checkpoint(
        model,
        optimizer,
        config.training.max_steps,
        f"{config.checkpoint.save_dir}/final.pt"
    )
            
if __name__ == "__main__":
    main()
    

    
    