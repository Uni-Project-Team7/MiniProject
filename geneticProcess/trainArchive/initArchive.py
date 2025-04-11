import sys
import os
sys.path.append(os.path.abspath("../../"))
from customOperations.sampler import CustomIntegerRandomSampling
from pymoo.core.problem import Problem
import json

if __name__ == '__main__' :
    problem = Problem(n_var=12, n_obj=2, xl=[0, 0, 0, 0, -2, -2, -2, -2, -2, -2, -2, -2], xu=[8, 8, 8, 8, -2, -2, -2, -2, -2, -2, -2, -2])
    sampler = CustomIntegerRandomSampling()
    pop = sampler._do(problem, 100)
    archive = []
    for i in pop:
        a = {'dv' : i.tolist(), 'L1Train' : -2.0, 'PSNRVal' : -2.0, 'Synflow' : -2.0, 'FLOPS' : -2, 'Params' : -2.0} 
        archive.append(a)

    with open('archive.json', 'w') as f:
        for item in archive:
            json.dump(item, f, separators=(', ', ': '))
            f.write('\n')
