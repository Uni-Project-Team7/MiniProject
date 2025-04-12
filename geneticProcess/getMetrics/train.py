import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models


def trainer(model, dataloader, device):
    model = model.to(device)
    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for blurred, sharp in dataloader:
            blurred, sharp = blurred.to(device), sharp.to(device)
            optimizer.zero_grad()
            output = model(blurred)
            loss = criterion(output, sharp)
            loss.backward()
            optimizer.step()

            batch_size = blurred.size(0)
            total_loss += loss.item() * batch_size  
            total_samples += batch_size

        epoch_loss = total_loss / total_samples
        

    return epoch_loss