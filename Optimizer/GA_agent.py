# GA_agent.py

import numpy as np
import random
from optimizer.fitness_evaluator import evaluate_fitness

# one-hot
nucleotides = ['A', 'T', 'G', 'C']
nucleotide_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}

class GA_Agent:
    def __init__(self, mutation_rate, predictor, device):
        self.mutation_rate = mutation_rate  # Mutation probability
        self.predictor = predictor  # Predictor model
        self.device = device

    def one_hot_encode(self, sequence):
        """one-hot"""
        return np.array([nucleotide_dict[nuc] for nuc in sequence])

    def one_hot_decode(self, encoded_sequence):
        """Convert one-hot encoding back to ATGC sequenc"""
        decoded_sequence = []
        for one_hot in encoded_sequence:
            index = np.argmax(one_hot)
            decoded_sequence.append(nucleotides[index])
        return ''.join(decoded_sequence)

    def mutate(sequence, mutation_rate):
        """
        Mutation operation: Perform random mutations on each base in the sequence according to the given mutation rate.
        """
        sequence_list = list(sequence)
        for i in range(len(sequence_list)):
            if random.random() < mutation_rate:
                sequence_list[i] = random.choice(['A', 'T', 'G', 'C'])
        return ''.join(sequence_list)

    def run_GA(self, population, center_value,mutation_rate):
        """
    Iteration: Traverse the population, perform mutations to generate offspring,
    update fitness using the predictor and evaluator, and select individuals with higher fitness.

    Parameters:
    - population: The current population, formatted as
    [{'sequence': sequence, 'expression': expression level, 'fitness': fitness}, ...]
    - center_value: The target central value of the expression level.

    Return value:
    - The new population, formatted as [{'sequence': sequence, 'expression': expression level, 'fitness': fitness}, ...]
"""

        new_population = []

        for parent in population:
            parent_sequence = parent['sequence']
            parent_expression = parent['expression']
            parent_fitness = parent['fitness']

            # 1. Perform mutation on the parent generation to generate offspring
            child_sequence = GA_Agent.mutate(parent_sequence, mutation_rate)

            # 2. Store the offspring in the offspring population list, with the fitness value initialized to None
            child_population = [{'sequence': child_sequence, 'expression': None, 'fitness': None}]

            # 3. Use the predictor to obtain the expression level of the offspring
            child_population = self.predictor.pre_seqs(child_population)
            # print(f"child_population after predictor: {child_population}")

            # 4. Call the evaluator to calculate the fitness value of the offspring
            child_population = evaluate_fitness(child_population, center_value)

            # Obtain the expression level and fitness of the offspring
            child_expression = child_population[0]['expression']
            child_fitness = child_population[0]['fitness']

            # 5. Compare the parent and offspring, and select the individual with higher fitness
            if child_fitness > parent_fitness:
                new_population.append({'sequence': child_sequence, 'expression': child_expression, 'fitness': child_fitness})
            else:
                new_population.append({'sequence': parent_sequence, 'expression': parent_expression, 'fitness': parent_fitness})

        return new_population

    def genetic_algorithm(self, population, mutation_rate, predictor):
        population_size = len(population)
        new_population = []
        child_population = []

        # Traverse each individual and perform mutation
        for i in range(population_size):
            parent = population[i]['sequence']

            child = GA_Agent.mutate(parent, mutation_rate)

            child_population.append({'sequence': child, 'expression': None})

        # Calculate the fitness values of the offspring population
        child_population = predictor.pre_seqs(child_population)  # 获取子代适应度值并更新种群
        # child_population = predictor.pre_seqs(child_population)[1]

        # Traverse the initial population and mutants, and perform fitness comparison
        for i in range(population_size):
            parent_fitness = population[i]['expression']
            child_fitness = child_population[i]['expression']

            # Select individuals with higher fitness and add them to the new population
            if child_fitness > parent_fitness:
                new_population.append(child_population[i])
            else:
                new_population.append(population[i])


            # if child_fitness < parent_fitness:
            #     new_population.append(child_population[i])
            # else:
            #     new_population.append(population[i])

        # Return the new population, including the optimized sequences and fitness values
        return new_population
