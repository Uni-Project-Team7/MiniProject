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
    #train_dataset = DeblurringDataset(dataset_type = 1)
    #test_dataset = DeblurringDataset(dataset_type = 0)
    #train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4)
    #test_dataloader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=4)

    torch.cuda.reset_peak_memory_stats()
    model = decode_and_build_unet(gene)
    model.to(device)

    #flops, params = get_flops(model)
    #start_train_time = time.time()
    #train_loss = trainer(model, train_dataloader, device)
    #end_train_time = time.time()
    #train_time = end_train_time - start_train_time

    #start_val_time = time.time()
    #psnr, ssim = evaluate_model_psnr(model, test_dataloader, device)
    #end_val_time = time.time()
    #val_time = end_val_time - start_val_time

    #candidate['flops'] = flops
    #candidate['params'] = params
    input_tensor = torch.randn(2, 3, 256, 256).to(device)
    target_tensor = torch.randn(2, 3, 256, 256).to(device)

    criterion = nn.L1Loss()

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    output = model(input_tensor)
    loss = criterion(output, target_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    max_mem = torch.cuda.max_memory_allocated()
    #candidate['psnr'] = psnr
    #candidate['train_time'] = train_time
    #candidate['val_time'] = val_time
    #candidate['ssim'] = ssim
    candidate['mem'] = max_mem








