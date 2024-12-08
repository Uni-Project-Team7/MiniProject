def bitflip_mutation_dynamic(Xp, mutation_prob=0.1):
    mutated_offsprings = Xp.copy()
    n_offsprings, n_var = Xp.shape
    n_model_keys = 4  #First 4 genes are model keys

    for i in range(n_offsprings):
        # Mutate the model keys (first 4 genes)
        for j in range(n_model_keys):
            if np.random.random() < mutation_prob:
                xl, xu = problem.xl[j], problem.xu[j]
                mutated_offsprings[i, j] = np.random.randint(xl, xu + 1)
              
        model_keys = mutated_offsprings[i, :n_model_keys].astype(int)

        # Get dynamic bounds for parameters
        dynamic_xl, dynamic_xu = dynamic_xu_xl(model_keys)

        # Mutate the parameters (remaining genes)
        for j, (xl, xu) in enumerate(zip(dynamic_xl, dynamic_xu), start=n_model_keys):
            if np.random.random() < mutation_prob:
                mutated_offsprings[i, j] = np.random.randint(xl, xu + 1)

    return mutated_offsprings

#Mutation probability
mutation_prob = 0.1

mutated_Xp = bitflip_mutation_dynamic(Xp, mutation_prob)

print("Mutated Offsprings:")
print(mutated_Xp)

