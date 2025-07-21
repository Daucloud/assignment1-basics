import torch
from torch import nn, sigmoid
from torch.nn.init import trunc_normal_
from einops import einsum, reduce, rearrange
from math import sqrt, ceil
from ..utils import attention

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None) -> None:
        super().__init__()
        sigma=sqrt(2/(in_features+out_features))
        self.W = nn.Parameter(torch.empty((out_features, in_features), dtype=dtype, device=device))
        trunc_normal_(tensor=self.W, std=sigma, a=-3*sigma, b=3*sigma)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "o i, ... i->... o")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) -> None:
        super().__init__()
        self.table = nn.Parameter(torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device))
        trunc_normal_(tensor=self.table, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor)->torch.Tensor:
        return self.table[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-5, device=None) -> None:
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.g = nn.Parameter(torch.ones((d_model), device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype=x.dtype
        x=x.to(torch.float32)
        rms=reduce(x, '... d -> ... 1', lambda x, axes: torch.sqrt(torch.mean(x**2, dim=axes)+self.eps))
        x=einsum(x, self.g, "... d, d -> ... d")/rms
        return x.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None) -> None:
        super().__init__()
        self.d_model=d_model
        self.d_ff=d_ff if d_ff else ceil(d_model/24)*64
        self.W1=Linear(self.d_model, self.d_ff, device=device)
        self.W2=Linear(self.d_ff, self.d_model, device=device)
        self.W3=Linear(self.d_model, self.d_ff, device=device)
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        W1x=self.W1(x)
        W3x=self.W3(x)
        sigmoid_W1x=W1x*sigmoid(W1x)
        return self.W2(sigmoid_W1x*W3x)

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        super().__init__()
        poses=torch.arange(max_seq_len, device=device).unsqueeze(1)
        dims=torch.arange(d_k//2, device=device).unsqueeze(0)
        angles=poses/(theta**(2*dims/d_k))
        self.register_buffer("cos_matrix", torch.cos(angles), persistent=False)
        self.register_buffer("sin_matrix", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x1, x2=x[...,::2], x[..., 1::2]
        rot_cos_mat=self.cos_matrix[token_positions]
        rot_sin_mat=self.sin_matrix[token_positions]
        rot_odd_x=x1*rot_cos_mat-x2*rot_sin_mat
        rot_even_x=x1*rot_sin_mat+x2*rot_cos_mat
        return rearrange([rot_odd_x, rot_even_x], "two ... d-> ... (d two)")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope=None, device=None) -> None:
        super().__init__()
        self.num_heads=num_heads
        self.QKV=Linear(d_model, 3*d_model, device=device)
        self.O=Linear(d_model, d_model, device=device)
        self.rope=rope
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None)->torch.Tensor:
        q, k, v=rearrange(self.QKV(x), "... s (three h d)-> three ... h s d", three=3, h=self.num_heads)
        if self.rope and token_positions is not None:
            q=self.rope(q, token_positions)
            k=self.rope(k, token_positions)
        mask=torch.tril(torch.ones(q.shape[-2],q.shape[-2], device=x.device)).bool()
        mh_qkv=rearrange(attention(q, k, v, mask), "... h s d -> ... s (h d)")
        return self.O(mh_qkv)

class Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff:int, rope: RoPE, device=None):
        super().__init__()
        self.norm1=RMSNorm(d_model=d_model, device=device)
        self.MHA=MultiHeadSelfAttention(d_model, num_heads, rope, device=device)
        self.norm2=RMSNorm(d_model=d_model, device=device)
        self.FFN=SwiGLU(d_model, d_ff, device=device)
    
    def forward(self, x: torch.Tensor, token_positions):
        x1=self.MHA(self.norm1(x), token_positions)+x
        return self.FFN(self.norm2(x1))+x1

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, d_ff:int, eps: float, theta: float, num_heads, device=None) -> None:
        super().__init__()
        self.embedding=Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device)
        self.rope=RoPE(theta, d_model//num_heads, context_length, device=device)
        self.blocks=nn.ModuleList(Block(d_model=d_model, num_heads=num_heads, rope=self.rope, device=device, d_ff=d_ff) for _ in range(num_layers))
        self.norm=RMSNorm(d_model, eps, device=None)
        self.linear=Linear(d_model, vocab_size, device=device)
    
    def forward(self, x, token_positions):
        x=self.embedding(x)
        for block in self.blocks:
            x=block(x, token_positions)
        return self.linear(self.norm(x))