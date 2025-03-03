import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train(model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, trainloader: torch.utils.data.DataLoader, epochs: int):
    model.train()
    loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=False)
    for epoch in range(epochs):
        for i, (imgs, targets) in loop:
            optimizer.zero_grad()
            y_pred = model(imgs)
            loss = criterion(y_pred, targets)
            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(loss=loss.item())
