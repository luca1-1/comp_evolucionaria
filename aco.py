import numpy as np
import random
import matplotlib.pyplot as plt
import time

def alg_aco(candidatos, population_size=50, num_iterations=400, alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100, plot_details=False):
    """
    Executa a seleção de características usando a Otimização por Colônia de Formigas (ACO) para encontrar o melhor subconjunto de características
    maximamente correlacionadas com uma variável alvo.
    """
    tempo_execucao = time.time()
    
    # Prepara os dados, removendo colunas não necessárias
    data = candidatos.drop(columns=['season', 'player', 'pos', 'team_id', 'award_share']).values
    target = candidatos['award_share'].values

    num_features = data.shape[1]
    
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

    # Inicializa o feromônio
    pheromone = np.ones(num_features)

    plot_data = []
    global_best_solution = None
    global_best_score = 0
    melhor_geracao = 0

    # ACO - Iterações
    for iteration in range(num_iterations):
        solutions = []
        scores = []

        for _ in range(population_size):
            solution = np.zeros(num_features, dtype=int)
            selected_features = np.random.choice(num_features, 10, replace=False)
            solution[selected_features] = 1
            
            # Calcula a aptidão
            score = fitness_function(solution, data, target)
            solutions.append(solution)
            scores.append(score)

            # Atualiza a melhor solução global
            if score > global_best_score:
                global_best_solution = solution
                global_best_score = score
                melhor_geracao = iteration

        # Atualiza o feromônio
        pheromone *= (1 - evaporation_rate)
        for solution, score in zip(solutions, scores):
            for i in range(num_features):
                if solution[i] == 1:
                    pheromone[i] += Q * score

        plot_data.append(global_best_score)

    # Obtém os nomes das características
    feature_names = candidatos.drop(columns=['season', 'player', 'pos', 'team_id', 'award_share']).columns
    selected_features = [feature_names[i] for i, gene in enumerate(global_best_solution) if gene == 1]

    # Ordena as melhores características
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

    return (selected_features, global_best_score)
