from ptflops import get_model_complexity_info


def get_flops(model):
    flops, params = get_model_complexity_info(model, (3, 512, 512), as_strings=False, print_per_layer_stat=False)
    return flops, params
    