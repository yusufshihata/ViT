import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.utils import save_checkpoint

def train(model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, trainloader: torch.utils.data.DataLoader, epochs: int = 30, log_interval: int = 10):
    model.train()
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    min_loss = float('inf')
    for epoch in range(epochs):
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        loop = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch+1}/{epochs}")
        for i, (imgs, targets) in loop:
            imgs, targets = imgs.to("cuda"), targets.to("cuda")
            optimizer.zero_grad()
            y_pred = model(imgs)
            loss = criterion(y_pred, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = y_pred.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            train_acc = 100 * train_correct / train_total
            
            if i % log_interval == 0 or i == len(trainloader) - 1:
                loop.set_postfix({
                    'loss': train_loss / (i + 1),
                    'acc': f"{train_acc:.2f}%"
                })
        
        if loss.item() < min_loss:
                min_loss = loss.item()
                save_checkpoint(model, optimizer, f"/kaggle/working/checkpoint_{epoch}.pth")
        
        epoch_train_loss = train_loss / len(trainloader)
        epoch_train_acc = 100 * train_correct / len(trainloader)
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
