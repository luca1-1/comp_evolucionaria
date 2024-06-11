import numpy as np
import random
import matplotlib.pyplot as plt
import time

def hybrid_algorithm(candidatos, population_size=50, num_iterations=100, inertia_weight=0.5, cognitive_weight=2.0, social_weight=2.0, mutation_rate=0.01, plot_details=False):
    tempo_execucao = time.time()
    data = candidatos.drop(columns=['season', 'player', 'pos', 'team_id', 'award_share']).values
    target = candidatos['award_share'].values

    def correlation(x, y):
        return np.corrcoef(x, y)[0, 1]

    def fitness_function(chromosome, data, target):
        selected_features = [i for i, gene in enumerate(chromosome) if gene == 1]
        if len(selected_features) == 0 or sum(chromosome) != 10:
            return 1e-6
        selected_data = data[:, selected_features]
        correlations = [abs(correlation(selected_data[:, i], target)) for i in range(selected_data.shape[1])]
        return sum(correlations) if correlations else 1e-6

    def initialize_population(population_size, chromosome_length, num_active_genes):
        population = []
        chromosome = np.zeros(chromosome_length, dtype=int)
        chromosome[:num_active_genes] = 1
        population.extend([np.random.permutation(chromosome).tolist() for _ in range(population_size)])
        return population

    def tournament_selection(population, fitnesses, k=3):
        if all(f == 0 for f in fitnesses):
            return random.choice(population)
        selected = random.choices(population, k=k, weights=fitnesses)
        return max(selected, key=lambda chrom: fitness_function(chrom, data, target))

    def one_point_crossover(parent1, parent2):
        point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2

    def mutate(chromosome, mutation_rate):
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        return chromosome

    plot_data = []
    melhor_fitness_geral = 0
    melhor_geracao = 0

    population = initialize_population(population_size, data.shape[1], 10)
    velocities = [np.random.uniform(-1, 1, size=data.shape[1]) for _ in range(population_size)]

    personal_best_positions = population.copy()
    personal_best_scores = [fitness_function(p, data, target) for p in population]
    global_best_position = max(personal_best_positions, key=lambda p: fitness_function(p, data, target))
    global_best_score = fitness_function(global_best_position, data, target)

    for iteration in range(num_iterations):
        fitnesses = [fitness_function(chrom, data, target) for chrom in population]
        for i in range(population_size):
            r1 = random.random()
            r2 = random.random()
            cognitive_component = cognitive_weight * r1 * (np.array(personal_best_positions[i]) - np.array(population[i]))
            social_component = social_weight * r2 * (np.array(global_best_position) - np.array(population[i]))
            velocities[i] = inertia_weight * np.array(velocities[i]) + cognitive_component + social_component

            population[i] = np.clip(np.array(population[i]) + velocities[i], 0, 1)
            population[i] = np.round(population[i]).astype(int)

            score = fitness_function(population[i], data, target)

            if score > personal_best_scores[i]:
                personal_best_positions[i] = population[i].copy()
                personal_best_scores[i] = score
        
        # Nova população
        new_population = []
        while len(new_population) < population_size:
            # Seleção
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            # Cruzamento
            child1, child2 = one_point_crossover(parent1, parent2)
            
            # Mutação
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]   

        current_global_best_position = max(personal_best_positions, key=lambda p: fitness_function(p, data, target))
        current_global_best_score = fitness_function(current_global_best_position, data, target)

        if current_global_best_score > global_best_score:
            global_best_position = current_global_best_position.copy()
            global_best_score = current_global_best_score
            melhor_geracao = iteration

        plot_data.append(global_best_score)

    best_chromosome = max(population, key=lambda chrom: fitness_function(chrom, data, target))

    feature_names = candidatos.drop(columns=['season', 'player', 'pos', 'team_id', 'award_share']).columns
    selected_features = [feature_names[i] for i, gene in enumerate(best_chromosome) if gene == 1]

    selected_features = sorted(selected_features, key=lambda x: abs(correlation(data[:, feature_names.get_loc(x)], target)), reverse=True)

    if plot_details:
        print(f"Tempo de execução: {time.time() - tempo_execucao}")
        print("Melhor aptidão:", global_best_score)
        print("Melhores atributos:")
        print(selected_features)
        print("Geração:", melhor_geracao)

        plt.plot(range(len(plot_data)), plot_data)
        plt.xlabel('Iteração')
        plt.ylabel('Fitness')
        plt.title('Evolução do Fitness ao Longo das Iterações')
        plt.show()

    return selected_features

