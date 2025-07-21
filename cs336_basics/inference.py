from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.traceback import install
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from .utils import load_checkpoint, softmax, set_seed
from .model.blocks import Transformer
from .tokenizer.tokenizer import Tokenizer
from .training.optimizer import AdamW

console=Console()
install()
config_name="config"

def generate(model, tokenizer, prompt, max_token, temperature=0.7, p=1, endtoken="<|endoftext|>"):
    with torch.no_grad():
        token_ids=torch.tensor(tokenizer.encode(prompt), device=model.embedding.table.device)
        prompt_len=len(token_ids)
        end_token_id=tokenizer.encode(endtoken)[0]
        while len(token_ids)<max_token:
            token_positions=torch.arange(len(token_ids), device=model.embedding.table.device)
            logits,_=torch.sort(model(token_ids, token_positions)[...,-1]/temperature, descending=True)
            probs=softmax(logits)
            cum_probs=torch.cumsum(probs, dim=-1)
            remove_indices=cum_probs>p
            remove_indices[...,1:]=remove_indices[...,:-1].clone()
            remove_indices[...,0]=False
            probs[remove_indices]=0
            output_token_id=torch.multinomial(probs, 1)
            token_ids=torch.cat([token_ids, output_token_id])
            if output_token_id is end_token_id:
                break
        return tokenizer.decode(token_ids.tolist()), tokenizer.decode(token_ids[prompt_len:].tolist())

@hydra.main(config_path="conf", config_name=config_name, version_base=None)
def main(cfg: DictConfig):
    console.print(Panel.fit(f"[bold green]Loading config {config_name}[/bold green]"))
    console.print(Panel.fit(OmegaConf.to_yaml(cfg)))
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
    model.eval()
    opt=AdamW(model.parameters(),lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay, betas=cfg.optimizer.betas, eps=cfg.optimizer.eps)
    load_checkpoint(cfg.inference.load_path, model, opt)
    tokenizer=Tokenizer.from_files(Path(cfg.tokenizer.path)/"vocab.json", Path(cfg.tokenizer.path)/"merges.txt", special_tokens=["<|endoftext|>"])
    while True:
        prompt=console.input("[bold red]Please input your prompt: [/bold red]")
        _, response=generate(model, tokenizer, prompt, cfg.inference.max_token, cfg.inference.temperature, cfg.inference.p)
        console.print(f"[bold green]{response}[/bold green]")

if __name__=='__main__':
    main()