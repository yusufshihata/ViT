import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.utils import calc_acc

def train(model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, trainloader: torch.utils.data.DataLoader, epochs: int):
    model.train()
    loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=False)
    for epoch in range(epochs):
        correct, total = 0, 0
        for i, (imgs, targets) in loop:
            optimizer.zero_grad()
            y_pred = model(imgs)
            loss = criterion(y_pred, targets)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(y_pred, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            acc = calc_acc(total, correct)

            loop.set_description(f"Epoch {epoch+1}/{epochs}")
            loop.set_postfix(loss=loss.item(), acc=acc*100)
