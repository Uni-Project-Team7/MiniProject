import numpy as np

def validate_and_adjust_offspring(offspring, xl, xu):
    """
    Validate and adjust the offspring encoding to meet semantic constraints.
    Increment parameters by 1 for duplicates while keeping them within bounds.

    Parameters:
    - offspring: List[int] - The 12-bit encoding array.
    - xl: np.array - Lower bounds for parameters (from dynamic_xu_xl).
    - xu: np.array - Upper bounds for parameters (from dynamic_xu_xl).

    Returns:
    - validated_offspring: List[int] - Adjusted encoding array.
    """
    models = offspring[:4]  # Models for the 4 levels
    parameters = np.array(offspring[4:]).reshape(4, 2)  # Parameters for each level

    # Iterate through each unique model and check for duplicates
    for model in set(models):
        indices = [i for i, m in enumerate(models) if m == model]
        if len(indices) > 1:  # If the model is present in multiple levels
            # Sort indices to process lower levels first
            indices.sort()

            # Validate and adjust parameters for duplicates
            for i in range(len(indices) - 1):
                lower_level, higher_level = indices[i], indices[i + 1]

                # Get bounds for the current model
                model_xu_start = 2 * model  # Start index of the model in bounds
                model_xu_end = model_xu_start + 2
                upper_bound = xu[model_xu_start:model_xu_end]
                lower_bound = xl[model_xu_start:model_xu_end]

                # Increment higher level parameters while ensuring constraints
                for p in range(2):  # For each parameter (2 parameters per model)
                    while parameters[higher_level][p] <= parameters[lower_level][p]:
                        parameters[higher_level][p] += 1
                        # Respect the bounds
                        if parameters[higher_level][p] > upper_bound[p]:
                            parameters[higher_level][p] = upper_bound[p]
                            break
                    # If the upper bound is 0 (special case), directly cap
                    if upper_bound[p] == 0:
                        parameters[higher_level][p] = 0

    # Reconstruct the validated offspring
    validated_offspring = list(offspring[:4]) + list(parameters.flatten())
    return validated_offspring

# Example Usage
def dynamic_xu_xl(model_keys):
    assert len(model_keys) == 4, "model_keys must contain exactly 4 values."
    xu = []
    xl = []
    for model_key in model_keys:
        match model_key:
            case 0:
                xl.extend([2, 0])
                xu.extend([16, 0])
            case 1:
                xl.extend([4, 1])
                xu.extend([8, 8])
            case 2:
                xl.extend([1, 0])
                xu.extend([28, 0])
            case 3:
                xl.extend([2, 2])
                xu.extend([6, 3])
            case 4:
                xl.extend([4, 0])
                xu.extend([16, 0])
            case 5:
                xl.extend([1, 0])
                xu.extend([28, 0])
            case 6:
                xl.extend([4, 1])
                xu.extend([8, 8])
            case 7:
                xl.extend([2, 2])
                xu.extend([16, 16])
            case 8:
                xl.extend([4, 0])
                xu.extend([8, 0])
    return np.array(xl), np.array(xu)

# Example input
offspring = [2, 2, 2, 2, 4, 3, 8, 0, 5, 2, 6, 4]  # Example input
model_keys = offspring[:4]
xl, xu = dynamic_xu_xl(model_keys)  # Generate bounds

validated_offspring = validate_and_adjust_offspring(offspring, xl, xu)
print("Input Offspring:", offspring)
print("Valid Offspring:", validated_offspring)
