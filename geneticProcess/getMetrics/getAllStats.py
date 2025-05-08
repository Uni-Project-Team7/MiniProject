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
    train_dataset = DeblurringDataset(blurred_dir='/workspace/nas_dataset/Datasets/train_pro/input_crops', sharp_dir='/workspace/nas_dataset/Datasets/train_pro/target_crops')
    test_dataset = DeblurringDataset(blurred_dir='/workspace/nas_dataset/Datasets/testpro/input_crops', sharp_dir='/workspace/nas_dataset/Datasets/testpro/target_crops')
    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=4)

    torch.cuda.reset_peak_memory_stats()
    model = decode_and_build_unet(gene)
    model.to(device)

    flops, params = get_flops(model)
    start_train_time = time.time()
    train_loss = trainer(model, train_dataloader, device)
    end_train_time = time.time()
    train_time = end_train_time - start_train_time

    start_val_time = time.time()
    psnr, ssim = evaluate_model_psnr(model, test_dataloader, device)
    end_val_time = time.time()
    val_time = end_val_time - start_val_time

    max_mem = torch.cuda.max_memory_allocated()
    candidate['flops'] = flops
    candidate['params'] = params
    candidate['train_loss'] = train_loss

    candidate['psnr'] = psnr
    candidate['train_time'] = train_time
    candidate['val_time'] = val_time
    candidate['ssim'] = ssim
    candidate['mem'] = max_mem








