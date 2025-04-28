# fitness_evaluator.py

def evaluate_fitness(population, center_value):
    """
    Evaluate the fitness of the entire population based on the predicted expression levels,
    with all results converging towards a central value.

    Parameters:
    - population: The population, formatted as
    [{'sequence': sequence, 'expression': expression level, 'fitness': None}, ...]
    - center_value: The target central value used to calculate fitness.

    Returns:
    - updated_population: The population with updated fitness, formatted as
    [{'sequence': sequence, 'expression': expression level, 'fitness': fitness}, ...]
"""

    updated_population = []

    for individual in population:
        # Obtain sequence and predicted expression levels
        sequence = individual['sequence']
        predicted_expression = individual['expression']

        # Calculate fitness based on the distance to the central value
        fitness = 1.0 / (abs(predicted_expression - center_value) + 1)

        # Update the fitness of the individual
        updated_population.append({'sequence': sequence, 'expression': predicted_expression, 'fitness': fitness})

    return updated_population
