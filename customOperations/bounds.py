import yaml
import numpy as np
import os


def dynamic_xu_xl(model_keys):
    assert len(model_keys) == 4
    config_path = os.path.join(os.path.dirname(__file__), "configs.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    xu = []
    xl = []
    for model_key in model_keys:
        model_name = config['model_conf'][model_key]
        xl.extend([config['xl'][model_name]['param1'], config['xl'][model_name]['param2']])
        xu.extend([config['xu'][model_name]['param1'], config['xu'][model_name]['param2']])

    return np.array(xl), np.array(xu)

if __name__ == "__main__" :
    print(dynamic_xu_xl([1,2,3,4]))
    