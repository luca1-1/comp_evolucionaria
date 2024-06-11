import numpy as np
import random
import matplotlib.pyplot as plt
import time

def alg_genetico(candidatos, population_size=50, num_generations=400, mutation_rate=0.01, num_crossover_points=1, plot_details=False):
    """
    Executa a seleção de características usando um Algoritmo Genético (GA) para encontrar o melhor subconjunto de características
    maximamente correlacionadas com uma variável alvo.
    """
    tempo_execucao = time.time()
    
    # Prepara os dados, removendo colunas não necessárias
    data = candidatos.drop(columns=['season', 'player', 'pos', 'team_id', 'award_share']).values
    target = candidatos['award_share'].values
    
    # Função para calcular a correlação entre dois atributos
    def correlation(x, y):
        return np.corrcoef(x, y)[0, 1]

    # Função de aptidão baseada na correlação máxima
    def fitness_function(chromosome, data, target):
        selected_features = [i for i, gene in enumerate(chromosome) if gene == 1]
        if len(selected_features) == 0 or sum(chromosome) != 10:
            return 1e-6  # Valor pequeno positivo ao invés de zero
        selected_data = data[:, selected_features]
        correlations = [abs(correlation(selected_data[:, i], target)) for i in range(selected_data.shape[1])]
        return sum(correlations) if correlations else 1e-6  # Valor pequeno positivo ao invés de zero
    
    # Inicializa uma população
    def initialize_population(population_size, chromosome_length, num_active_genes):
        """
        Inicializa uma população com um número exato de genes ativos por cromossomo.

        Args:
            population_size (int): O tamanho da população.
            chromosome_length (int): O comprimento de cada cromossomo (número total de genes).
            num_active_genes (int): O número de genes ativos (definidos como 1) em cada cromossomo.

        Returns:
            list: A população inicializada.
        """
        population = []
        chromosome = np.zeros(chromosome_length, dtype=int)
        chromosome[:num_active_genes] = 1
        population.extend([np.random.permutation(chromosome).tolist() for _ in range(population_size)])
        return population

    population = initialize_population(population_size, data.shape[1], 10)

    # Função de seleção por torneio
    def tournament_selection(population, fitnesses, k=3):
        if all(f == 0 for f in fitnesses):
            return random.choice(population)
        selected = random.choices(population, k=k, weights=fitnesses)
        return max(selected, key=lambda chrom: fitness_function(chrom, data, target))
    
    # Função de cruzamento de múltiplos pontos
    def multi_point_crossover(parent1, parent2, num_points):
        points = sorted(random.sample(range(1, len(parent1)), num_points))
        child1, child2 = parent1.copy(), parent2.copy()
        for i in range(len(points)):
            if i % 2 == 0:
                if i == len(points) - 1:
                    child1[points[i]:], child2[points[i]:] = parent2[points[i]:], parent1[points[i]:]
                else:
                    child1[points[i]:points[i+1]], child2[points[i]:points[i+1]] = parent2[points[i]:points[i+1]], parent1[points[i]:points[i+1]]
            else:
                if i == len(points) - 1:
                    child1[points[i]:], child2[points[i]:] = parent2[points[i]:], parent1[points[i]:]
                else:
                    child1[points[i]:points[i+1]], child2[points[i]:points[i+1]] = parent2[points[i]:points[i+1]], parent1[points[i]:points[i+1]]
        return child1, child2

    # Função de mutação
    def mutate(chromosome, mutation_rate):
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        return chromosome

    plot_data = []
    melhor_fitness_geral = 0
    melhor_geracao = 0
    
    # GA - Gerações
    for generation in range(num_generations):
        # Avaliar aptidão da população
        fitnesses = [fitness_function(chrom, data, target) for chrom in population]
        
        melhor_fitness_atual = max(fitnesses)
        plot_data.append(melhor_fitness_atual)
        if melhor_fitness_atual > melhor_fitness_geral:
            melhor_fitness_geral = melhor_fitness_atual
            melhor_geracao = generation
        
        # Nova população
        new_population = []
        while len(new_population) < population_size:
            # Seleção
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            # Cruzamento
            child1, child2 = multi_point_crossover(parent1, parent2, num_crossover_points)
            
            # Mutação
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]

    # Encontrar o melhor cromossomo na população final
    best_chromosome = max(population, key=lambda chrom: fitness_function(chrom, data, target))

    # Obter os nomes das características
    feature_names = candidatos.drop(columns=['season', 'player', 'pos', 'team_id', 'award_share']).columns
    selected_features = [feature_names[i] for i, gene in enumerate(best_chromosome) if gene == 1]

    # Ordenar os 10 melhores atributos
    selected_features = sorted(selected_features, key=lambda x: abs(correlation(data[:, feature_names.get_loc(x)], target)), reverse=True)
    
    if plot_details:
        print(f"Tempo de execução: {time.time() - tempo_execucao}")
        print("Melhor aptidão:", melhor_fitness_geral)
        print("Melhores atributos:")
        print(selected_features)
        print("Geração:", melhor_geracao)

        plt.plot(range(len(plot_data)), plot_data)
        plt.xlabel('Geração')
        plt.ylabel('Fitness')
        plt.title('Evolução do Fitness ao Longo das Gerações')
        plt.show()

    return selected_features
