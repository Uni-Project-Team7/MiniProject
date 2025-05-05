import sys
import os
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath("../../"))
from customOperations.archBuilderDir.encodingToArch import decode_and_build_unet
from geneticProcess.getMetrics.dataloader import DeblurringDataset
import time
from geneticProcess.getMetrics.FLOPSandParams import get_flops
from geneticProcess.getMetrics.train import trainer
from geneticProcess.getMetrics.valPSNR import evaluate_model_psnr
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.functional import peak_signal_noise_ratio
from torchmetrics.functional import structural_similarity_index_measure

def get_stats(candidate, device):

    gene = candidate['gene']
    print(f'evaluating gene : {gene}')


    torch.cuda.reset_peak_memory_stats()
    model = decode_and_build_unet(gene)
    model.to(device)

    flops, params = get_flops(model)
    print(flops)
    print(params)
    start_train_time = time.time()
    train_loss = 0
    end_train_time = time.time()
    train_time = end_train_time - start_train_time

    start_val_time = time.time()
    psnr, ssim = 0, 0
    end_val_time = time.time()
    val_time = end_val_time - start_val_time

    input_tensor = torch.rand((2, 3, 64, 64))
    ground_truth = torch.rand((2, 3, 64, 64))
    input_tensor = input_tensor.to(device)
    ground_truth = ground_truth.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    output = model(input_tensor)

    criterion = nn.L1Loss()
    loss = criterion(output, ground_truth)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    psnr_val = peak_signal_noise_ratio(output, ground_truth, data_range=1.0)
    ssim_val = structural_similarity_index_measure(output, ground_truth, data_range=1.0)
    max_mem = torch.cuda.max_memory_allocated()
    candidate['flops'] = flops
    candidate['params'] = params
    candidate['train_loss'] = train_loss

    candidate['psnr'] = psnr_val.item()
    #candidate['train_time'] = train_time
    #candidate['val_time'] = val_time
    candidate['ssim'] = ssim_val.item()
    candidate['mem'] = max_mem








