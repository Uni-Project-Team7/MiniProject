from ptflops import get_model_complexity_info
import torch

def get_flops(model):
    model.eval()
    with torch.no_grad() :
        flops, params = get_model_complexity_info(model, (3, 512, 512), as_strings=False, print_per_layer_stat=False)
    return flops, params
    