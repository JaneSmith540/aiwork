import numpy as np
import random
import copy
from typing import List, Tuple, Callable


class GeneticAlgorithmTSP:
    def __init__(self, distance_matrix, population_size=100,
                 generations=500, crossover_rate=0.8,
                 mutation_rate=0.2, elitism_rate=0.1):
        """
        初始化遗传算法参数
        """
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = int(population_size * elitism_rate)

        # 最佳解记录
        self.best_individual = None
        self.best_fitness = float('inf')
        self.fitness_history = []

    def create_individual(self) -> List[int]:
        """创建随机个体（路径）"""
        individual = list(range(self.num_cities))
        random.shuffle(individual)
        return individual

    def create_population(self) -> List[List[int]]:
        """创建初始种群"""
        return [self.create_individual() for _ in range(self.population_size)]

    def calculate_fitness(self, individual: List[int]) -> float:
        """计算个体的适应度（路径长度）"""
        total_distance = 0
        for i in range(self.num_cities):
            from_city = individual[i]
            to_city = individual[(i + 1) % self.num_cities]
            total_distance += self.distance_matrix[from_city][to_city]
        return total_distance

    def tournament_selection(self, population: List[List[int]],
                             fitnesses: List[float], k: int = 3) -> List[int]:
        """锦标赛选择"""
        selected = random.sample(list(zip(population, fitnesses)), k)
        selected.sort(key=lambda x: x[1])
        return copy.deepcopy(selected[0][0])

    def order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """顺序交叉（OX）"""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        size = len(parent1)
        # 随机选择两个交叉点
        start, end = sorted(random.sample(range(size), 2))

        # 创建子代
        child1 = [-1] * size
        child2 = [-1] * size

        # 复制中间段
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]

        # 填充剩余位置
        self._fill_remaining_genes(child1, parent2, start, end)
        self._fill_remaining_genes(child2, parent1, start, end)

        return child1, child2

    def _fill_remaining_genes(self, child: List[int], parent: List[int],
                              start: int, end: int):
        """填充剩余基因"""
        size = len(child)
        parent_index = 0

        for i in range(size):
            position = (end + i) % size

            if child[position] == -1:
                while parent[parent_index % size] in child:
                    parent_index += 1
                child[position] = parent[parent_index % size]
                parent_index += 1

    def swap_mutation(self, individual: List[int]) -> List[int]:
        """交换变异"""
        if random.random() > self.mutation_rate:
            return individual

        mutated = copy.deepcopy(individual)
        i, j = random.sample(range(len(individual)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated

    def inversion_mutation(self, individual: List[int]) -> List[int]:
        """逆转变异"""
        if random.random() > self.mutation_rate:
            return individual

        mutated = copy.deepcopy(individual)
        i, j = sorted(random.sample(range(len(individual)), 2))
        mutated[i:j + 1] = reversed(mutated[i:j + 1])
        return mutated

    def run(self):
        """运行遗传算法"""
        # 初始化种群
        population = self.create_population()

        for generation in range(self.generations):
            # 计算适应度
            fitnesses = [self.calculate_fitness(ind) for ind in population]

            # 更新最佳解
            min_fitness = min(fitnesses)
            min_index = fitnesses.index(min_fitness)

            if min_fitness < self.best_fitness:
                self.best_fitness = min_fitness
                self.best_individual = copy.deepcopy(population[min_index])

            self.fitness_history.append(self.best_fitness)

            # 选择精英
            elite_indices = np.argsort(fitnesses)[:self.elitism_count]
            new_population = [copy.deepcopy(population[i]) for i in elite_indices]

            # 生成新种群
            while len(new_population) < self.population_size:
                # 选择
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)

                # 交叉
                child1, child2 = self.order_crossover(parent1, parent2)

                # 变异
                if random.random() < 0.5:
                    child1 = self.swap_mutation(child1)
                    child2 = self.swap_mutation(child2)
                else:
                    child1 = self.inversion_mutation(child1)
                    child2 = self.inversion_mutation(child2)

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

            # 每100代打印进度
            if generation % 100 == 0:
                print(f"Generation {generation}: Best Fitness = {self.best_fitness:.2f}")

        return self.best_individual, self.best_fitness, self.fitness_history