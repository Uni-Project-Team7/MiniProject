import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath("../../"))
from customOperations.archBuilderDir.encodingToArch import decode_and_build_unet
from geneticProcess.getMetrics.dataloader import DeblurringDataset


torch.backends.cudnn.benchmark = True


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train(model, dataloader, criterion, optimizer, device):
    num_epochs = 6

    for epoch in range(num_epochs):
        model.train()
        for blurred, sharp in dataloader:
            blurred, sharp = blurred.to(device), sharp.to(device)
            optimizer.zero_grad()
            output = model(blurred)
            loss = criterion(output, sharp)
            loss.backward()
            optimizer.step()


def inter(rank:int, world_size: int, gene):
    ddp_setup(rank, world_size)
    device = rank

    model = decode_and_build_unet(gene).to(device)
    model = DDP(model, device_ids=[rank])

    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    batch_size = 4
    workers = 22

    train_dataset = DeblurringDataset(dataset_type = 1)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, sampler=DistributedSampler(train_dataset))

    train(model, train_dataloader, criterion, optimizer, device)

    if rank == 0:
        checkpoint = {
            'model': model.module.state_dict(),
        }
        torch.save(checkpoint, './temp/check.pth')

    destroy_process_group()

def trainer(gene):
    world_size = torch.cuda.device_count()
    mp.spawn(inter, args=(world_size, gene), nprocs=world_size)
