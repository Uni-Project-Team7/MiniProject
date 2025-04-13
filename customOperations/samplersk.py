import sys
import os
import yaml
import numpy as np
from pymoo.core.individual import Individual
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.problem import Problem

# Add the customOperations folder to the Python path
sys.path.append('/teamspace/studios/this_studio/MiniProject')

from customOperations.bounds import dynamic_xu_xl  # Now this should work

class CustomIntegerRandomSampling(FloatRandomSampling):
    def _do(self, problem, n_samples, rng, **kwargs):
        # Ensure we use rng for reproducibility
        n = int(problem.n_var / 3)
        xl, xu = problem.bounds()
        xl = xl[0:n]
        xu = xu[0:n]

        # Load the configuration from the inline config (just for this case)
        config_path = os.path.join(os.path.dirname(__file__), "configs.yaml")  # Corrected path
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        # Use rng to sample models, generating n_samples random model indices
        X = np.column_stack([rng.integers(config["model_key_xl"], config["model_key_xu"], size=n_samples) for _ in range(n)])
        full_solution = np.zeros((n_samples, problem.n_var))
        full_solution[:, :n] = X

        # Generate the dynamic parameters for each sample
        for i in range(n_samples):
            Dxl, Dxu = dynamic_xu_xl(X[i])
            dynamic_params = np.array([rng.integers(Dxl[k], Dxu[k] + 1) for k in range(2 * n)])
            ans = self.sample_param1(rng)  # Pass rng to sample_param1 for parameter generation
            for j in range(4):
                dynamic_params[j * 2] = ans[j]
            
            # Store the dynamic parameters back into the solution
            full_solution[i, n:] = dynamic_params

        # Create individuals from the full solution
        individuals = np.array([Individual(X=x) for x in full_solution])
        return full_solution

    def sample_param1(self, rng):
        # This function generates incremental parameters with rng
        ans = []
        max_bound = [4, 8, 12, 16]
        prev = 1
        for i in range(4):
            prev = rng.integers(prev, max_bound[i] + 1)  # Using rng for reproducibility
            ans.append(prev)

        return ans

if __name__ == '__main__':
    # Set a fixed seed for rng to ensure reproducibility
    seed = 42
    rng = np.random.default_rng(seed)

    # Create the problem object
    problem = Problem(n_var=12, n_obj=2, xl=[0, 0, 0, 0, -2, -2, -2, -2, -2, -2, -2, -2], xu=[8, 8, 8, 8, -2, -2, -2, -2, -2, -2, -2, -2])

    # Create the custom sampler and generate the population
    sampler = CustomIntegerRandomSampling()
    pop1 = sampler._do(problem, 30, rng)  # Pass rng here
    print(pop1)
