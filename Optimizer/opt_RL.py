# opt_RL.py

import torch
import torch.optim as optim
from optimizer.actor_critic import ActorCritic
from optimizer.fitness_evaluator import evaluate_fitness
from optimizer.predictor import PREDICT
from optimizer.GA_agent import GA_Agent
import numpy as np
import csv


class Optimizer:
    def __init__(self):

        self.predictor_model_path = "/Predictor/results/model_SC_short/LSTMModel_SC_short.pth"
        self.population_file_path = "/Generator/samples/SC_short/G_2999.txt"
        self.save_population_path = "/optimizer/results_opt/SC_80bp/medium_final_population_SC.csv"
        self.save_sequences_path = "/optimizer/results_opt/SC_80bp/medium_final_sequences_SC.txt"
        self.model_save_path = "/optimizer/results_opt/SC_80bp/medium_actor_critic_model.pth"
        self.natural_sequence_path = "/Data/SC/SC_seq_short.txt"

        self.generations = 50
        self.mutation_rate = 0.1
        self.gamma = 0.99
        self.hidden_size = 64
        # self.action_size = 2
        self.action_size = 3
        self.center_value = 10
        self.center_range = (-2,2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  initial population
    def load_initial_population(self):
        population = []
        with open(self.population_file_path, 'r') as f:
            for line in f:
                sequence = line.strip()
                population.append({'sequence': sequence, 'expression': None})
        return population

    def save_population(self, population):
        with open(self.save_population_path, 'w') as f:
            writer = csv.writer(f)
            for ind in population:
                writer.writerow([ind['sequence'], ind['expression'], ind['fitness']])
                # writer.writerow([ind['sequence'], ind['expression']])

    def save_sequences(self, population):
        with open(self.save_sequences_path, 'w') as f:
            for ind in population:
                f.write(f"{ind['sequence']}\n")

    # save model
    def save_models(self, actor_critic):
        torch.save(actor_critic.state_dict(), self.model_save_path)
        # torch.save(predictor.model.state_dict(), self.predictor_save_path)
        print(f"save model: Actor-Critic -> {self.model_save_path}")

    def calculate_kmer_frequency(self,sequence, k):
        kmer_count = {}
        total_kmers = 0

        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i + k]
            if kmer in kmer_count:
                kmer_count[kmer] += 1
            else:
                kmer_count[kmer] = 1
            total_kmers += 1

        kmer_frequency = {kmer: count / total_kmers for kmer, count in kmer_count.items()}
        return kmer_frequency

    def calculate_pearson_correlation(self, seq1, seq2, k=4):
        freq1 = self.calculate_kmer_frequency(seq1, k)
        freq2 = self.calculate_kmer_frequency(seq2, k)

        common_kmers = set(freq1.keys()) & set(freq2.keys())

        if len(common_kmers) == 0:
            return 0.0

        x = np.array([freq1[kmer] for kmer in common_kmers])
        y = np.array([freq2[kmer] for kmer in common_kmers])

        correlation = np.corrcoef(x, y)[0, 1]
        return correlation

    def compare_sequence_files(self, gen_sequence, natural_sequence, k=4):
        # k-mer
        correlation = self.calculate_pearson_correlation(gen_sequence, natural_sequence, k)
        return correlation

    # update Actor-Critic model
    def update_actor_critic(self, actor_critic, optimizer, states, actions, rewards, next_states, dones, gamma):
        action_probs, state_values = actor_critic(states)
        _, next_state_values = actor_critic(next_states)

        td_error = rewards + (1 - dones) * gamma * next_state_values - state_values
        critic_loss = td_error.pow(2).mean()

        actor_loss = -(torch.log(action_probs.gather(1, actions.unsqueeze(1))) * td_error.detach()).mean()

        loss = actor_loss + critic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # select action
    def select_action(self, actor_critic, state):
        state = state.to(self.device)
        action_probs, _ = actor_critic(state)
        action = torch.multinomial(action_probs, 1).item()  # 根据动作概率分布选择动作
        return action

    # train Actor-Critic and optimize GA parameters
    def train_actor_critic_with_ga(self, generation, ga_agent, actor_critic, optimizer, natural_sequence, population, predictor):
        # evaluate the fitness of the current population
        expression_value = [ind['expression'] for ind in population]
        expression_mean = np.mean(expression_value)
        fitness_values = [ind['fitness'] for ind in population]
        fitness_mean = np.mean(fitness_values)

        sequences = [ind['sequence'] for ind in population]
        gen_sequence = ''.join(sequences)
        k_mer = self.compare_sequence_files(gen_sequence, natural_sequence)

        print(f"Generation {generation}: Mean fitness = {fitness_mean}, Mean expression = {expression_mean}, 4-mer = {k_mer}")
        # print(f"Generation {generation}: Mean expression = {expression_mean}, 4-mer = {k_mer}")

        # Train the reinforcement learning model using state information
        state = torch.tensor([fitness_mean, expression_mean, k_mer], dtype=torch.float32).to(self.device)
        # state = torch.tensor([ expression_mean, k_mer], dtype=torch.float32).to(self.device)
        action = self.select_action(actor_critic, state)  # select action

        # Action: Adjust the mutation rate
        if action == 0:
            self.mutation_rate += 0.01
        elif action == 1:
            self.mutation_rate -= 0.01

        # Execute the genetic algorithm using GA_Agent
        new_population = ga_agent.run_GA(population, self.center_value, self.mutation_rate)
        # new_population = ga_agent.genetic_algorithm(population, self.mutation_rate, predictor)

        # Calculate the reward
        new_fitness_values = [ind['fitness'] for ind in new_population]
        new_fitness_mean = np.mean(new_fitness_values)

        new_expression_value = [ind['expression'] for ind in new_population]
        next_expression_mean = np.mean(new_expression_value)

        new_sequences = [ind['sequence'] for ind in new_population]
        new_gen_sequence = ''.join(new_sequences)
        new_k_mer = self.compare_sequence_files(new_gen_sequence, natural_sequence)

        # reward = 0.5 * (next_expression_mean - expression_mean) + 0.5 * (new_k_mer - k_mer + 0.05)
        reward = 0.5 * (new_fitness_mean - fitness_mean) + 0.5 * (new_k_mer - k_mer + 0.05)
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)

        # Get the new state and update the Actor-Critic model
        next_state = torch.tensor([new_fitness_mean, next_expression_mean, new_k_mer], dtype=torch.float32).to(self.device)
        # next_state = torch.tensor([next_expression_mean, new_k_mer], dtype=torch.float32).to(self.device)
        done = torch.tensor([0], dtype=torch.float32).to(self.device)

        self.update_actor_critic(actor_critic, optimizer, state.unsqueeze(0), torch.tensor([action]).to(self.device),
                                 reward, next_state.unsqueeze(0), done, self.gamma)

        return new_population, self.mutation_rate


def main():

    torch.cuda.set_device(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    opt = Optimizer()
    predictor = PREDICT(opt.predictor_model_path)
    actor_critic = ActorCritic(opt.hidden_size, opt.action_size).to(opt.device)
    optimizer = optim.Adam(actor_critic.parameters(), lr=0.001)
    ga_agent = GA_Agent(mutation_rate=opt.mutation_rate, predictor=predictor, device=device)
    population = opt.load_initial_population()

    with open(opt.natural_sequence_path, 'r') as natural_file:
        natural_sequence = natural_file.read().strip()

    population = predictor.pre_seqs(population)
    population = evaluate_fitness(population, opt.center_value)

    for generation in range(opt.generations):
        population, mutation_rate = opt.train_actor_critic_with_ga(
            generation, ga_agent, actor_critic, optimizer,
            natural_sequence, population, predictor)

    opt.save_population(population)
    opt.save_sequences(population)
    opt.save_models(actor_critic)


if __name__ == "__main__":
    main()
