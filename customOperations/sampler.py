from pymoo.core.individual import Individual
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.problem import Problem
from customOperations.bounds import dynamic_xu_xl
import numpy as np
import yaml
import os


class CustomIntegerRandomSampling(FloatRandomSampling):
    # def __init__(self, problem, n_samples, **kwargs):
    #     super().__init__()

    def _do(self, problem, n_samples, **kwargs):
        n = int(problem.n_var / 3)
        xl, xu = problem.bounds()
        xl = xl[0:n]
        xu = xu[0:n]
        config_path = os.path.join(os.path.dirname(__file__), "configs.yaml")
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        X = np.column_stack([np.random.randint(config["model_key_xl"], config["model_key_xu"], size=n_samples) for _ in range(n)])
        full_solution = np.zeros((n_samples, problem.n_var))
        full_solution[:, :n] = X
        for i in range(n_samples):
            Dxl, Dxu = dynamic_xu_xl(X[i])
            dynamic_params = np.array([np.random.randint(Dxl[k], Dxu[k] + 1) for k in range(2*n)])
            ans = self.sample_param1()
            for j in range(4) :
                dynamic_params[j * 2] = ans[j]
            full_solution[i, n:] = dynamic_params


        individuals = np.array([Individual(X=x) for x in full_solution])
        return full_solution
    
    def sample_param1(self):
        ans = []
        max_bound = [4, 8, 12, 16]
        prev = 1
        for i in range(4):
            prev = np.random.randint(prev, max_bound[i] + 1)
            ans.append(prev)
        
        return ans


if __name__ == '__main__' :

    problem = Problem(n_var=12, n_obj=2, xl=[0, 0, 0, 0, -2, -2, -2, -2, -2, -2, -2, -2], xu=[8, 8, 8, 8, -2, -2, -2, -2, -2, -2, -2, -2])
    sampler = CustomIntegerRandomSampling()
    pop1 = sampler._do(problem, 30)
    print(pop1)