import time
from functools import partial
from pathlib import Path
from turtle import st
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from .training.loss import cross_entropy
from .utils import gradient_clip, cosine_anneal, load_checkpoint, train_data_load, save_checkpoint, set_seed, eval_data_load
import wandb
from .model.blocks import Transformer
from .training.optimizer import AdamW
from rich.console import Console
from rich.traceback import install
from rich.panel import Panel
from rich.progress import Progress
from .tokenizer.tokenizer import Tokenizer

console=Console()
install()

config_name="config"

def preprocess(tokenizer, data_path, save_path):
    with open(data_path) as f:
        token_ids=np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    np.save(save_path, token_ids)

def train_one_step(model: torch.nn.Module, opt: torch.optim.Optimizer, it:int, input: torch.Tensor, targets, M, lr_max, lr_min, T_w, T_c):
    opt.zero_grad()
    token_positions=torch.arange(input.shape[-1], device=input.device)
    logits=model(input, token_positions)
    loss=cross_entropy(logits, targets)
    wandb.log({"train/loss": loss.item()}, step=it)
    loss.backward()
    gradient_clip(model.parameters(), M)
    new_lr=cosine_anneal(it, lr_max, lr_min, T_w, T_c)
    for group in opt.param_groups:
        group['lr']=new_lr
    opt.step()
    return loss

def evaluate(model: torch.nn.Module, partial_eval_data_load, progress:Progress, task):
    with torch.no_grad():
        model.eval()
        losses=[]
        step=0
        while True:
            pair=partial_eval_data_load(step=step)
            step+=1
            if pair is None:
                break
            x,y=pair
            token_positions=torch.arange(x.shape[-1], device=model.embedding.table.device)
            loss=cross_entropy(model(x,token_positions),y)
            losses.append(loss)
            progress.update(task, advance=1, description=f"[bold yellow] Evaulating... Loss {loss.item(): .4f}")
        result=torch.exp(torch.mean(torch.tensor(losses)))
        model.train()
        return result

@hydra.main(config_path="conf", config_name=config_name, version_base=None)
def train_loop(cfg: DictConfig):
    console.print(Panel.fit(f"[bold green]Loading config {config_name}[/bold green]"))
    console.print(Panel.fit(OmegaConf.to_yaml(cfg)))
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    set_seed(cfg.seed)
    model=Transformer(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        num_layers=cfg.model.num_layers,
        d_model=cfg.model.d_model,
        d_ff=cfg.model.d_ff,
        eps=cfg.model.eps,
        theta=cfg.model.theta,
        num_heads=cfg.model.num_heads,
        device=cfg.device
    ).to(cfg.device)
    opt=AdamW(model.parameters(),lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay, betas=cfg.optimizer.betas, eps=cfg.optimizer.eps)
    tokenizer=Tokenizer.from_files(Path(cfg.tokenizer.path)/"vocab.json", Path(cfg.tokenizer.path)/"merges.txt", special_tokens=["<|endoftext|>"])
    if cfg.train.load_path is not None:
        load_checkpoint(cfg.train.load_path, model, opt)
    try:
        train_data=np.memmap(cfg.data.preprocessed_train_data_path)
    except:
        console.print("[bold red]Preprocessing training data started[bold red]")
        preprocess(tokenizer, cfg.data.raw_train_data_path, cfg.data.preprocessed_train_data_path)
        console.print("[bold red]Preprocessing training ended[bold red]")
        train_data=np.memmap(cfg.data.preprocessed_train_data_path)
    if cfg.train.eval_steps>0:
        try:
            eval_data=np.memmap(cfg.data.preprocessed_eval_data_path)
        except:
            console.print("[bold red]Preprocessing evaluation data started[bold red]")
            preprocess(tokenizer, cfg.data.raw_eval_data_path, cfg.data.preprocessed_eval_data_path)
            console.print("[bold red]Preprocessing evaluation ended[bold red]")
            eval_data=np.memmap(cfg.data.preprocessed_eval_data_path)
    set_seed(cfg.seed)
    with Progress() as progress:
        task=progress.add_task("[bold cyan] Training...", total=cfg.train.steps)
        last_ppl=None
        for it in range(1, cfg.train.steps+1):
            if cfg.train.eval_steps>0 and it%cfg.train.eval_steps==0:
                eval_task=progress.add_task("[bold yellow]Evaluating...", total=len(eval_data)//(cfg.train.batch_size*cfg.model.context_length))
                partial_eval_data_load=partial(eval_data_load,x=eval_data, batch_size=cfg.train.batch_size, context_length=cfg.model.context_length, device=cfg.device)
                last_ppl=evaluate(model,partial_eval_data_load, progress, eval_task)
                wandb.log({"eval/perplexity": last_ppl})
                progress.remove_task(eval_task)
            x,y=train_data_load(train_data, cfg.train.batch_size, cfg.model.context_length, cfg.device)
            loss=train_one_step(model, opt, it, x, y, cfg.train.clip_value, cfg.optimizer.lr, 0, cfg.train.warmup_ratio*cfg.train.steps, cfg.train.steps)
            progress.update(task, advance=1, description=f"[bold cyan] Training... Loss: {loss.item():.4f}"+(f"| Eval PPL: {last_ppl.item(): .4f}" if last_ppl is not None else ""))

    Path(cfg.train.save_path).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, opt, it, cfg.train.save_path)
    wandb.finish()

if __name__=='__main__':
    train_loop()