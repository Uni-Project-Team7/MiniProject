import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm

def trainer(model, dataloader, device):
    model = model.to(device)
    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        # Add tqdm here
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for blurred, sharp in pbar:
            blurred, sharp = blurred.to(device), sharp.to(device)
            optimizer.zero_grad()
            output = model(blurred)
            loss = criterion(output, sharp)
            loss.backward()
            optimizer.step()

            batch_size = blurred.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            avg_loss = total_loss / total_samples
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

        epoch_loss = total_loss / total_samples
        print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {epoch_loss:.4f}")

    return epoch_loss


