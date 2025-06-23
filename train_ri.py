# train_ri.py
# Trains R_i using previously stored ρ_i and a balanced dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np

def train_ri(model, trainset, class_idx, rho_i, complement_size=5, device='cuda'):
    positive_idx = [i for i, (_, label) in enumerate(trainset) if label == class_idx]
    negative_classes = np.argsort(rho_i)[-complement_size:]  # top ρi classes
    negative_idx = [i for i, (_, label) in enumerate(trainset) if label in negative_classes]

    indices = positive_idx + negative_idx
    loader = DataLoader(Subset(trainset, indices), batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()
    for epoch in range(10):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model
