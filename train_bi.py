# train_bi.py
# Trains B_i classifiers and stores the ρ_i confidence vectors

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
import pickle

def train_bi(model, trainset, class_idx, device='cuda'):
    indices = [i for i, (_, label) in enumerate(trainset) if label == class_idx]
    loader = DataLoader(Subset(trainset, indices), batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()
    for epoch in range(10):  # example epoch count
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model

def collect_rho_i(model, val_loader, device='cuda'):
    model.eval()
    confidence = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            outputs = torch.softmax(model(inputs), dim=1)
            confidence.append(outputs.cpu().numpy())
    return np.concatenate(confidence, axis=0).mean(axis=0)  # average confidence vector
