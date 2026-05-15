from collections.abc import Callable, Iterable
from typing import Optional
import math
import torch

def cross_entropy(pred: torch.Tensor, target: torch.Tensor) -> float:
    shifted_logits = pred - torch.max(pred, dim=-1, keepdim=True).values
    
    logsumexp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1, keepdim=True))
    log_probs = shifted_logits - logsumexp
    
    target_log_probs = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    return -torch.mean(target_log_probs)

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
                return loss
            
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if(len(state) == 0):
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                
                t = state.get("t", 0)
                grad = p.grad.data
                lr_t = lr * math.sqrt(1 - beta2 ** (t + 1)) / (1 - beta1 ** (t + 1))
                p.data -= lr * weight_decay * p.data  # Apply weight decay.
                state["m"] = beta1 * state["m"] + (1 - beta1) * grad
                state["v"] = beta2 * state["v"] + (1 - beta2) * grad ** 2
                p.data -= lr_t * state["m"] / (torch.sqrt(state["v"]) + eps)  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss
    
def learning_rate_schedule(t: int, lr_max: float, lr_min: float, warmup_iters: int, total_iters: int) -> float:
    if t < warmup_iters:
        return lr_max * t / warmup_iters
    elif t <= total_iters:
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (t - warmup_iters) / (total_iters - warmup_iters)))
    else:
        return lr_min
    
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_norm: float):
    eps = 1e-6
    
    params = [p for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(sum(p.grad.data.norm(2) ** 2 for p in params))
    
    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1:
        for p in params:
            p.grad.data.mul_(clip_coef)
            
    return total_norm