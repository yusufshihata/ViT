import torch

def calc_acc(num_pred: int, num_true: int) -> float:
    return num_pred / num_true
