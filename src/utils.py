import torch

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, path)

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer
