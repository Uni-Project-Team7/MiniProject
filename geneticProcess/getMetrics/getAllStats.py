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

def get_stats(candidate):

    gene = candidate['gene']
    print(f'evaluating gene : {gene}')
    torch.cuda.reset_peak_memory_stats()
    model = decode_and_build_unet(gene)
    model.to('cuda:0')
    batch_size = 6

    flops, params = get_flops(model)
    start_train_time = time.time()
    trainer(gene, batch_size)
    end_train_time = time.time()
    train_time = end_train_time - start_train_time

    start_val_time = time.time()
    psnr, ssim = evaluate_model_psnr(gene, batch_size)
    end_val_time = time.time()
    val_time = end_val_time - start_val_time

    candidate['flops'] = flops
    candidate['params'] = params
    candidate['psnr'] = psnr
    candidate['ssim'] = ssim
    candidate['train_time'] = train_time
    candidate['val_time'] = val_time
    candidate['mem'] = max_mem








