import contextvars
import numpy as np
import torch
import math
from einops import einsum
from math import sqrt

PRETOKENIZE_PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def get_new_string(old_string:list[bytes], merged_pair:bytes)->list[bytes]:
    new_string=[]
    i=0
    while i<len(old_string):
        if i<len(old_string)-1 and old_string[i]+old_string[i+1]==merged_pair:
            new_string.append(merged_pair)
            i+=2
        else:
            new_string.append(old_string[i])
            i+=1
    return new_string

def softmax(x: torch.Tensor, dim: int=-1):
    x_max, _=torch.max(x, dim=dim, keepdim=True)
    x=x-x_max
    return torch.exp(x)/torch.sum(torch.exp(x), dim=dim, keepdim=True)

def attention(Q, K, V, mask=None):
    x=einsum(Q, K, "... n d_k, ... m d_k -> ... n m")
    x=x/sqrt(K.shape[-1])
    if mask is not None:
        x+=torch.where(mask, 0.0, -torch.inf)
    return einsum(softmax(x, -1), V, "... n m, ... m d_v -> ... n d_v")

def cosine_anneal(t,lr_max, lr_min, T_w, T_c):
    if t<T_w:
        return t*lr_max/T_w
    elif T_w<=t<=T_c:
        return lr_min+0.5*(1+math.cos(math.pi*((t-T_w)/(T_c-T_w))))*(lr_max-lr_min)
    else:
        return lr_min

def gradient_clip(params, M, eps=1e-6):
    cnt=0
    for p in params:
        if p.grad is None:
            continue
        cnt+=torch.sum(p.grad**2)
    clip_ratio=M/(torch.sqrt(cnt)+eps)
    if clip_ratio<1.0:
        for p in params:
            if p.grad is None:
                continue
            p.grad*=clip_ratio

def train_data_load(x, batch_size, context_length, device=None):
    start_indices=torch.randint(0,len(x)-context_length, (batch_size,))
    offset=torch.arange(context_length)
    indices=start_indices.unsqueeze(1)+offset
    data=torch.tensor(x[indices],device=device, dtype=torch.long)
    targets=torch.tensor(x[indices+1],device=device, dtype=torch.long)
    return data, targets

def eval_data_load(x, batch_size, context_length, step, device=None):
    start_index=step*batch_size*context_length
    if start_index+batch_size*context_length > len(x)-1:
        return None
    batch_starts=torch.arange(start_index, start_index+batch_size*context_length, context_length)
    offset=torch.arange(context_length)
    indices=batch_starts.unsqueeze(1)+offset
    data=torch.tensor(x[indices], device=device, dtype=torch.long)
    targets=torch.tensor(x[indices+1], device=device, dtype=torch.long)
    return data, targets

def save_checkpoint(model, optimizer, iteration, out):
    save_dict={
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(save_dict, out)

def load_checkpoint(src, model, optimizer):
    save_dict=torch.load(src)
    model.load_state_dict(save_dict["model"])
    optimizer.load_state_dict(save_dict["optimizer"])
    return save_dict["iteration"]

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)