import torch

def cross_entropy(logits, targets):
    logits_max, _=torch.max(logits, dim=-1, keepdim=True)
    logits=logits-logits_max
    log_p=torch.logsumexp(logits, dim=-1)-torch.gather(logits,-1,targets.unsqueeze(1))
    return torch.mean(log_p)