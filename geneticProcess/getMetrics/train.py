import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from time import time

def trainer(model, dataloader, device):
    model = model.to(device)
    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 1
    epoch_loss = 0
    start_t = time.time()
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

   print(start_time - time.time())


    return epoch_loss
