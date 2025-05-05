import sys
import os
sys.path.append(os.path.abspath("../../"))
from customOperations.sampler import CustomIntegerRandomSampling
from pymoo.core.problem import Problem
import json

if __name__ == '__main__' :
    problem = Problem(n_var=12, n_obj=2, xl=[0, 0, 0, 0, -2, -2, -2, -2, -2, -2, -2, -2], xu=[8, 8, 8, 8, -2, -2, -2, -2, -2, -2, -2, -2])
    sampler = CustomIntegerRandomSampling()
    pop = sampler._do(problem, 2000)
    archive = []
    for i in pop:
        a = {'gene' : i.tolist(), 'flops' : -2, 'params' : -2, 'train_loss' : -2, 'psnr' : -2, 'ssim' : -2, 'mem' : -2}
        archive.append(a)

    with open('archive.json', 'w') as f:
        for item in archive:
            json.dump(item, f, separators=(', ', ': '))
            f.write('\n')
