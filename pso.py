import numpy as np
import random
import matplotlib.pyplot as plt
import time

def alg_pso(candidatos, population_size=50, num_iterations=400, inertia_weight=0.5, cognitive_weight=2.0, social_weight=2.0, plot_details=False):
    """
    Executa a seleção de características usando a Otimização por Enxame de Partículas (PSO) para encontrar o melhor subconjunto de características
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
            return 0
        selected_data = data[:, selected_features]
        correlations = [abs(correlation(selected_data[:, i], target)) for i in range(selected_data.shape[1])]
        return sum(correlations) if correlations else 0

    # Inicializa uma população de partículas
    def initialize_particles(population_size, chromosome_length, num_active_genes):
        """
        Inicializa uma população de partículas com um número exato de genes ativos por partícula.

        Args:
            population_size (int): O tamanho da população.
            chromosome_length (int): O comprimento de cada cromossomo (número total de genes).
            num_active_genes (int): O número de genes ativos (definidos como 1) em cada cromossomo.

        Returns:
            list: A população inicializada.
        """
        particles = []
        chromosome = np.zeros(chromosome_length, dtype=int)
        chromosome[:num_active_genes] = 1
        particles.extend([np.random.permutation(chromosome).tolist() for _ in range(population_size)])
        return particles

    # Inicializar população (partículas) e velocidades
    particles = initialize_particles(population_size, data.shape[1], 10)
    velocities = [np.random.uniform(-1, 1, size=data.shape[1]) for _ in range(population_size)]

    # Inicializar melhores posições pessoais e global
    personal_best_positions = particles.copy()
    personal_best_scores = [fitness_function(p, data, target) for p in particles]
    global_best_position = max(personal_best_positions, key=lambda p: fitness_function(p, data, target))
    global_best_score = fitness_function(global_best_position, data, target)

    plot_data = []
    melhor_geracao = 0

    # PSO - Iterações
    for iteration in range(num_iterations):
        for i in range(population_size):
            # Atualizar velocidade
            r1 = random.random()
            r2 = random.random()
            cognitive_component = cognitive_weight * r1 * (np.array(personal_best_positions[i]) - np.array(particles[i]))
            social_component = social_weight * r2 * (np.array(global_best_position) - np.array(particles[i]))
            velocities[i] = inertia_weight * np.array(velocities[i]) + cognitive_component + social_component

            # Atualizar posição
            particles[i] = np.clip(np.array(particles[i]) + velocities[i], 0, 1)
            particles[i] = np.round(particles[i]).astype(int)

            # Garantir que exatamente 10 genes estejam ativos
            while sum(particles[i]) != 10:
                if sum(particles[i]) < 10:
                    zero_indices = [j for j in range(len(particles[i])) if particles[i][j] == 0]
                    particles[i][random.choice(zero_indices)] = 1
                elif sum(particles[i]) > 10:
                    one_indices = [j for j in range(len(particles[i])) if particles[i][j] == 1]
                    particles[i][random.choice(one_indices)] = 0

            # Calcular nova aptidão
            score = fitness_function(particles[i], data, target)

            # Atualizar melhor posição pessoal
            if score > personal_best_scores[i]:
                personal_best_positions[i] = particles[i].copy()
                personal_best_scores[i] = score

        # Atualizar melhor posição global
        current_global_best_position = max(personal_best_positions, key=lambda p: fitness_function(p, data, target))
        current_global_best_score = fitness_function(current_global_best_position, data, target)

        if current_global_best_score > global_best_score:
            global_best_position = current_global_best_position.copy()
            global_best_score = current_global_best_score
            melhor_geracao = iteration

        plot_data.append(global_best_score)

    # Obter os nomes das características
    feature_names = candidatos.drop(columns=['season', 'player', 'pos', 'team_id', 'award_share']).columns
    selected_features = [feature_names[i] for i, gene in enumerate(global_best_position) if gene == 1]

    # Ordenar os 10 melhores atributos
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
