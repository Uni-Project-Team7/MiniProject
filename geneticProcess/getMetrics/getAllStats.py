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

def get_stats(candidate, device):
    
    gene = candidate['gene']
    print(f'evaluating gene : {gene}')

    train_dataset = DeblurringDataset(blurred_dir='/teamspace/studios/this_studio/dataset/train_crops/blur_crops', sharp_dir='/teamspace/studios/this_studio/dataset/train_crops/sharp_crops')
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_dataset = DeblurringDataset(blurred_dir='/teamspace/studios/this_studio/dataset/val_crops/blur_crops', sharp_dir='/teamspace/studios/this_studio/dataset/val_crops/sharp_crops')
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    model = decode_and_build_unet(gene)
    model.to(device)
    
    flops, params = get_flops(model)
    
    start_train_time = time.time()
    train_loss = trainer(model, train_dataloader, device)
    end_train_time = time.time()
    train_time = end_train_time - start_train_time

    start_val_time = time.time()
    psnr = evaluate_model_psnr(model, test_dataloader, device)
    end_val_time = time.time()
    val_time = end_val_time - start_val_time

    candidate['flops'] = flops
    candidate['params'] = params
    candidate['train_loss'] = train_loss
    candidate['psnr'] = psnr
    candidate['train_time'] = train_time
    candidate['val_time'] = val_time
    
    







