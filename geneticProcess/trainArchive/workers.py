from torch.utils.data import DataLoader
import time
import os
import sys

sys.path.append(os.path.abspath("../../"))
from customOperations.archBuilderDir.encodingToArch import     decode_and_build_unet
from geneticProcess.getMetrics.dataloader import DeblurringDataset
def benchmark_dataloader(dataloader):
    start = time.time()
    for i, batch in enumerate(dataloader):
        if i == 100:  # Check time to load first 100 batches
            break
    end = time.time()
    return end - start

train = DeblurringDataset(dataset_type = 1)
for num_workers in range(0, os.cpu_count() + 1, 2):  # Try 0, 2, 4, ..., max
    dataloader = DataLoader(train, batch_size=4, num_workers=num_workers)
    duration = benchmark_dataloader(dataloader)
    print(f"num_workers={num_workers}: {duration:.2f}s")
