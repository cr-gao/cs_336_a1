import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        '''
        in_features: int  final dimension of the input
        out_features: int  final dimension of the output
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        '''
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        
        sigma = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a = -3.0 * sigma, b = 3.0 * sigma)  
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        '''
        num_embeddings: int  Size of the vocabulary
        embedding_dim: int  Dimension of the embedding vectors, i.e., d_model
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        '''
        super().__init__()
        
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
    
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        '''
        Lookup the embedding vectors for the given token IDs
        '''
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        '''
        Construct the RMSNorm module. This function should accept the following parameters:
        d_model: int  Hidden dimension of the model
        eps: float = 1e-5  Epsilon value for numerical stability
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        '''
        super().__init__()
        
        self.d_model = d_model
        self.eps = eps
        
        self.weight = nn.Parameter(torch.empty((d_model,), device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        '''
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        '''
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        rms = torch.sqrt((1.0 / self.d_model) * torch.sum(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight
        
        return result.to(in_dtype)
    
class SwiGLU_FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        '''
        Construct the SwiGLU_FFN module. This function should accept the following parameters:
        d_model: int  Hidden dimension of the model
        d_ff: int  Hidden dimension of the feedforward network
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        '''
        super().__init__()
        
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        '''
        gate = self.w1(x)
        gate = gate * torch.sigmoid(gate)
        
        values = self.w3(x)
        
        return self.w2(gate * values)
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        '''
        Construct the RoPE module and create buffers if needed.
        theta: float Θ value for the RoPE
        d_k: int  dimension of query and key vectors
        max_seq_len: int  Maximum sequence length that will be input
        device: torch.device | None = None  Device to store the buffer on
        '''
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        positions = torch.arange(max_seq_len, device=device)
        angles = positions[:, None] * inv_freq[None, :]
        
        self.register_buffer('cos_cached', torch.cos(angles), persistent=False)
        self.register_buffer('sin_cached', torch.sin(angles), persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note 
        that you should tolerate x with an arbitrary number of batch dimensions. You should assume 
        that the token positions are a tensor of shape (..., seq_len) specifying the token positions of x 
        along the sequence dimension.
        '''
        x = x.reshape(*x.shape[:-1], self.d_k // 2, 2)
        
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        x1 = x[..., 0] * cos - x[..., 1] * sin
        x2 = x[..., 0] * sin + x[..., 1] * cos
        
        new_x = torch.stack([x1, x2], dim=-1).reshape(*x.shape[:-2], self.d_k)
        return new_x

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    
    return x_exp / x_exp_sum

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k = q.shape[-1]
    pre_softmax = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(mask == 0, float('-inf'))
        
    attention_weights = softmax(pre_softmax, dim=-1)
    return attention_weights @ v

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_rope: bool, theta: float | None=None, max_seq_len: int | None=None, device=None, dtype=None):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta=theta or 10000.0, d_k=d_model // num_heads, max_seq_len=max_seq_len or 2048, device=device)
    
    def forward(self, 
                x: torch.Tensor, 
                token_positions: torch.Tensor, 
                past_k=None, 
                past_v=None, 
                use_cache=False) -> torch.Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        d_k = self.d_model // self.num_heads
        [B, T, _] = x.shape
        Q = Q.view(B, T, self.num_heads, d_k).transpose(1, 2).contiguous()
        K = K.view(B, T, self.num_heads, d_k).transpose(1, 2).contiguous()
        V = V.view(B, T, self.num_heads, d_k).transpose(1, 2).contiguous()
        
        if self.use_rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
            
        if past_k is not None and past_v is not None:
            K = torch.cat([past_k, K], dim=-2)
            V = torch.cat([past_v, V], dim=-2)
        
        causal_mask = torch.triu(torch.ones((T, T), device=x.device), diagonal=1).bool()
        causal_mask = ~causal_mask
        
        attention_output = scaled_dot_product_attention(Q, K, V, causal_mask)
        attention_output = attention_output.transpose(1, 2).reshape(B, T, self.d_model)
        
        output = self.o_proj(attention_output)

        if use_cache:
            return output, K, V
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, use_rope: bool, theta: float | None=None, max_seq_len: int | None=None, device=None, dtype=None):
        super().__init__()
        
        self.attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, use_rope=use_rope, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.ffn = SwiGLU_FFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.norm1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
    
    def forward(self, 
                x: torch.Tensor, 
                token_positions: torch.Tensor, 
                past_k=None, 
                past_v=None, 
                use_cache=False) -> torch.Tensor:
        if use_cache:
            attn_output, K, V = self.attention(self.norm1(x), token_positions, past_k=past_k, past_v=past_v, use_cache=use_cache)
        else:
            attn_output = self.attention(self.norm1(x), token_positions)
        x = x + attn_output
        
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        
        if use_cache:
            return x, K, V
        return x
    
class TransformerLM(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, context_size: int, num_layers: int, use_rope: bool, theta: float | None=None, max_seq_len: int | None=None, device=None, dtype=None):
        super().__init__()
        
        self.token_embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, use_rope=use_rope, theta=theta, max_seq_len=context_size, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.output_projection = Linear(d_model, vocab_size, device=device, dtype=dtype)
        
    def forward(self, 
                token_ids: torch.Tensor, 
                kv_cache=None, 
                use_cache=False) -> torch.Tensor:
        x = self.token_embedding(token_ids)
        
        if kv_cache is not None:
            assert use_cache, "kv_cache provided but use_cache is False"
            assert len(kv_cache) == len(self.layers), "Length of kv_cache must match number of layers"
            past_length = kv_cache[0][0].shape[-2]
            token_positions = torch.arange(past_length, past_length + token_ids.shape[-1], device=token_ids.device)
        else:
            token_positions = torch.arange(token_ids.shape[-1], device=token_ids.device)

        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            if kv_cache is not None:
                past_k, past_v = kv_cache[i]
                x, K, V = layer(x, token_positions, past_k=past_k, past_v=past_v, use_cache=use_cache)
                new_kv_cache.append((K, V))
            elif use_cache:
                x, K, V = layer(x, token_positions, use_cache=use_cache)
                new_kv_cache.append((K, V))
            else:
                x = layer(x, token_positions)
        
        x = self.norm(x)
        logits = self.output_projection(x)
        if use_cache:
            return logits, new_kv_cache
        return logits
        