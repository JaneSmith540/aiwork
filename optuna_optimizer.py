import optuna
import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithmTSP
from tsp_loader import load_tsp_file, calculate_distance_matrix


def objective(trial, distance_matrix):
    """
    Optuna目标函数
    """
    # 定义超参数搜索空间
    population_size = trial.suggest_int('population_size', 50, 300)
    generations = trial.suggest_int('generations', 200, 1000)
    crossover_rate = trial.suggest_float('crossover_rate', 0.6, 0.95)
    mutation_rate = trial.suggest_float('mutation_rate', 0.01, 0.3)
    elitism_rate = trial.suggest_float('elitism_rate', 0.05, 0.2)

    # 运行遗传算法
    ga = GeneticAlgorithmTSP(
        distance_matrix=distance_matrix,
        population_size=population_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elitism_rate=elitism_rate
    )

    _, best_fitness, _ = ga.run()

    return best_fitness


def optimize_parameters(distance_matrix, n_trials=50):
    """
    使用Optuna优化参数
    """
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # 包装目标函数
    func = lambda trial: objective(trial, distance_matrix)

    study.optimize(func, n_trials=n_trials, show_progress_bar=True)

    print("\n最佳参数:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"最佳适应度: {study.best_value:.2f}")

    return study