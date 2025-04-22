import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import sys
import os
sys.path.append(os.path.abspath("../../"))
from customOperations.archBuilderDir.encodingToArch import decode_and_build_unet
from geneticProcess.getMetrics.dataloader import DeblurringDataset

def evaluate_model_psnr(model, dataloader, device):
    model.eval()
    model.to(device)
    
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    
    with torch.no_grad():
        for blurred, sharp in tqdm(dataloader):
            blurred = blurred.to(device)
            sharp = sharp.to(device)
            
            deblurred = model(blurred)
            deblurred = torch.clamp(deblurred, 0, 1)
            
            psnr_value = peak_signal_noise_ratio(deblurred, sharp, data_range=1.0)
            ssim_value = structural_similarity_index_measure(deblurred, sharp, data_range=1.0)
            
            total_psnr += psnr_value.item() * blurred.size(0)
            total_ssim += ssim_value.item() * blurred.size(0)
            count += blurred.size(0)
    
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    
    return avg_psnr, avg_ssim

if __name__ == "__main__":

    # TODO: check datarange  = 1
    model = decode_and_build_unet([ 2,  2,  3,  1,  3,  0,  8,  0, 11,  0, 15,  2])
    dataset = DeblurringDataset(blurred_dir='/teamspace/studios/this_studio/dataset/train_crops/blur_crops', sharp_dir='/teamspace/studios/this_studio/dataset/train_crops/sharp_crops')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    avg_psnr, avg_ssim = evaluate_model_psnr(model, dataloader, 'cuda:0')
    print(f"Average PSNR: {avg_psnr:.2f} dB") 