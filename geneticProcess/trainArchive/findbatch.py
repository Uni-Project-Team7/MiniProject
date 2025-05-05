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
import traceback

def find_max_batch_size(model: nn.Module, device=None, image_sizes=[64, 128, 256, 384, 448, 512], max_batch=128):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.train()

    criterion = nn.L1Loss()

    results = {}

    for img_size in image_sizes:
        print(f"\nTesting image size: {img_size}x{img_size}")
        low = 0
        high = max_batch
        max_valid_batch = 0

        while low <= high:
            mid = (low + high) // 2
            try:
                # Clear GPU memory
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Generate dummy input and ground truth
                dummy_input = torch.randn(mid, 3, img_size, img_size, device=device, requires_grad=True)
                dummy_target = torch.randn_like(dummy_input)

                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

                optimizer.zero_grad()
                output = model(dummy_input)
                loss = criterion(output, dummy_target)
                loss.backward()
                optimizer.step()

                max_valid_batch = mid
                low = mid + 1

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    high = mid - 1
                else:
                    print(f"Unexpected error at batch size {mid}: {e}")
                    traceback.print_exc()
                    break

        results[img_size] = max_valid_batch
        print(f"Max trainable batch size for {img_size}x{img_size}: {max_valid_batch}")

    return results

gene = [2.0, 0.0, 5.0, 4.0, 1.0, 0.0, 1.0, 0.0, 12.0, 2.0, 15.0, 4.0]

model = decode_and_build_unet(gene).to('cuda:0')

results = find_max_batch_size(model)

print("\nSummary:")
for size, batch in results.items():
    print(f"Image size {size}x{size} -> Max batch size: {batch}")
