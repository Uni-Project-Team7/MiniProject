import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch.nn as nn
import torch.optim as optim
from torchmetrics import PeakSignalNoiseRatio
import torch.nn.functional as F
import torchvision.models as models
from thop import profile
from ..customOperations.archBuilder.encodingToArch import decode_and_build_unet

def get_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.2)])


class DeblurringDataset(Dataset):
    def __init__(self, blurred_dir, sharp_dir, transform=None):
        self.blurred_dir = blurred_dir
        self.sharp_dir = sharp_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(blurred_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        blurred_path = os.path.join(self.blurred_dir, img_name)
        sharp_path = os.path.join(self.sharp_dir, img_name)
        
        blurred = cv2.imread(blurred_path)
        blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        sharp = cv2.imread(sharp_path)
        sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=blurred, image0=sharp)
            blurred = augmented['image']
            sharp = augmented['image0']
        
        return blurred, sharp


def trainer(encoded_array) :
    blurred_dir = "path/to/blurred"
    sharp_dir = "path/to/sharp"

    # Create dataset
    transform = get_augmentations()
    dataset = DeblurringDataset(blurred_dir, sharp_dir, transform=transform)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)




    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = decode_and_build_unet(encoded_array, 64).cuda()
    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for blurred, sharp in dataloader:
            blurred, sharp = blurred.to(device), sharp.to(device)
            optimizer.zero_grad()
            output = model(blurred)
            loss = criterion(output, sharp)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    test_blurred_dir = "path/to/test/blurred"
    test_sharp_dir = "path/to/test/sharp"
    test_dataset = DeblurringDataset(test_blurred_dir, test_sharp_dir, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    psnr_metric = PeakSignalNoiseRatio().to(device)
    model.eval()
    psnr_total = 0.0
    with torch.no_grad():
        for blurred, sharp in test_dataloader:
            blurred, sharp = blurred.to(device), sharp.to(device)
            output = model(blurred)
            psnr_total += psnr_metric(output, sharp).item()

    psnr_avg = psnr_total / len(test_dataloader)
    print(f"Average PSNR: {psnr_avg:.2f} dB")


    input_sample = torch.randn(1, 3, 512, 512).to(device)
    profile_result = profile(model, inputs=(input_sample,))
    if isinstance(profile_result, tuple) and len(profile_result) == 2:
        flops, params = profile_result
    else:
        flops, params, _ = profile_result  
    print(f"FLOPs: {flops/1e9:.2f} GFLOPs, Parameters: {params/1e6:.2f}M")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'psnr': psnr_avg,
        'flops': flops
    }
    torch.save(checkpoint, "deblurring_checkpoint.pth")
