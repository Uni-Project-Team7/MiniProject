import numpy as np
def validate_offspring_with_repeated_models(offspring, dynamic_xu_xl):
    # Extract models and parameters
    model_keys = offspring[:4]
    parameters = offspring[4:]
  
    # Get dynamic bounds based on the model keys
    dynamic_xl, dynamic_xu = dynamic_xu_xl(model_keys)

    # Detect repeated models
    unique_models = {}
    for idx, model in enumerate(model_keys):
        if model not in unique_models:
            unique_models[model] = []
        unique_models[model].append(idx)

    for model, indices in unique_models.items():
        if len(indices) > 1:  # Only process if model is repeated
          
            repeated_params = [parameters[i * 2:(i + 1) * 2] for i in indices]

            # Compare and update parameters
            max_params = repeated_params[0]  
            for i in range(1, len(repeated_params)):
                current_params = repeated_params[i]
                updated_params = []

                for max_p, current_p, xl, xu in zip(max_params, current_params, dynamic_xl, dynamic_xu):
                    # If the current parameter is less than or equal to the max, update it
                    if current_p <= max_p:
                        updated_params.append(min(max_p + 1, xu))  # Ensure within bounds
                    else:
                        updated_params.append(current_p)

                # Update the offspring with the new parameters
                parameters[indices[i] * 2:(indices[i] + 1) * 2] = updated_params
                max_params = updated_params 

    return model_keys + parameters

# Round mutated offspring to the nearest integer
mutated_Xp = [np.round(offspring).astype(int).tolist() for offspring in mutated_Xp]
print("Rounded Mutated Offsprings:")
for offspring in mutated_Xp:
    print(offspring)

# semantic validation to the rounded offsprings
validated_offsprings = [validate_offspring_with_repeated_models(offspring, dynamic_xu_xl) for offspring in mutated_Xp]
print("\nValidated Offsprings after Rounding and Validation:")
for validated in validated_offsprings:
    print(validated)
