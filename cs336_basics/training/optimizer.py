from email.policy import default
from torch.optim import Optimizer
from collections.abc import Callable, Iterable
from typing import Any, Dict, Optional, Tuple
from torch import Tensor
import math
import torch

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss

class AdamW(Optimizer):
    def __init__(self, params, lr, betas, weight_decay, eps) -> None:
        defaults={
            "lr":lr,
            "beta1":betas[0],
            "beta2":betas[1],
            "decay":weight_decay,
            "eps":eps
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr,beta1,beta2,decay,eps=[group[k] for k in ["lr","beta1","beta2","decay","eps"]]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state=self.state[p]
                t=state.get("t",1)
                m=state.get("m",torch.zeros_like(p))
                v=state.get("v",torch.zeros_like(p))
                grad=p.grad.data
                m=beta1*m+(1-beta1)*grad
                v=beta2*v+(1-beta2)*grad**2
                lr_t=lr*math.sqrt(1-beta2**t)/(1-beta1**t)
                p.data-=lr_t*m/(torch.sqrt(v)+eps)
                p.data-=lr*decay*p.data
                state["t"]=t+1
                state["m"]=m
                state["v"]=v
        return loss

if __name__=='__main__':
    import matplotlib.pyplot as plt
    for lr, color, linestyle in zip([1e1, 1e2, 1e3], ['red', 'green', 'blue'], ['-', '--', ':']):
        torch.manual_seed(42)
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        y_data=[]
        opt = SGD([weights], lr=lr)
        for t in range(100):
            opt.zero_grad() # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean() # Compute a scalar loss value.
            print(loss.cpu().item())
            y_data.append(loss.cpu().item())
            loss.backward() # Run backward pass, which computes gradients.
            opt.step() # Run optimizer step.
        plt.plot(y_data, label=f'lr={lr}', color=color, linestyle=linestyle)
    plt.legend()
    plt.show()