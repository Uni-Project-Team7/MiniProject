import sys
import os
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath("../../"))
from customOperations.archBuilderDir.encodingToArch import decode_and_build_unet
from geneticProcess.getMetrics.dataloader import DeblurringDatasetTrain, DeblurringDatasetVal, flatten_patch_collate_fn
import time
from geneticProcess.getMetrics.FLOPSandParams import get_flops
from geneticProcess.getMetrics.train import trainer
from geneticProcess.getMetrics.valPSNR import evaluate_model_psnr
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.functional import peak_signal_noise_ratio
from torchmetrics.functional import structural_similarity_index_measure

def get_stats(candidate, device = 'cuda:0'):

    gene = candidate['gene']
    print(f'evaluating gene : {gene}')
    train_dataset = DeblurringDatasetTrain()
    test_dataset = DeblurringDatasetVal()
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=22, collate_fn=flatten_patch_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=22)

    torch.cuda.reset_peak_memory_stats()
    model = decode_and_build_unet(gene)
    model.to(device)

    flops, params = get_flops(model)
    print(f'FLOPs : {flops}')

    start_train_time = time.time()
    train_loss = trainer(model, train_dataloader, device)
    end_train_time = time.time()
    train_time = end_train_time - start_train_time

    start_val_time = time.time()
    psnr, ssim = evaluate_model_psnr(model, test_dataloader, device)
    end_val_time = time.time()
    val_time = end_val_time - start_val_time

    print(f'Val PSNR : {psnr}')
    print(f'Val SSIM : {ssim}')

    #input_tensor = torch.rand((1, 3, 256, 256))
    #ground_truth = torch.rand((1, 3, 256, 256))
    #input_tensor = input_tensor.to(device)
    #ground_truth = ground_truth.to(device)
    #optimizer = optim.SGD(model.parameters(), lr=0.01)
    #output = model(input_tensor)
    #criterion = nn.L1Loss()
    #loss = criterion(output, ground_truth)
    #optimizer.zero_grad()
    #loss.backward()
    #optimizer.step()
    #psnr_val = peak_signal_noise_ratio(output, ground_truth, data_range=1.0)
    #ssim_val = structural_similarity_index_measure(output, ground_truth, data_range=1.0)

    max_mem = torch.cuda.max_memory_allocated()
    candidate['flops'] = flops
    candidate['params'] = params
    candidate['train_loss'] = train_loss

    candidate['psnr'] = psnr
    candidate['train_time'] = train_time
    candidate['val_time'] = val_time
    candidate['ssim'] = ssim
    candidate['mem'] = max_mem














