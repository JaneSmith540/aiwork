import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice
import time
import warnings
import pandas as pd
import random
from typing import Dict, List, Tuple
import os
import json
from scipy.interpolate import griddata
from scipy.stats import norm

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class TSPSystematicOptimizer:
    """TSP遗传算法系统化优化器（Optuna优化模块）"""

    def __init__(self, distance_matrix, cities):
        self.distance_matrix = distance_matrix
        self.cities = cities
        self.best_params = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.optimization_history = []

    def objective_function(self, trial):
        """Optuna目标函数：最小化路径长度"""
        # 定义搜索空间
        params = {
            'population_size': trial.suggest_int('population_size', 50, 500),
            'crossover_rate': trial.suggest_float('crossover_rate', 0.6, 0.95),
            'mutation_rate': trial.suggest_float('mutation_rate', 0.01, 0.5),
            'elitism_rate': trial.suggest_float('elitism_rate', 0.05, 0.3),
            'generations': trial.suggest_int('generations', 200, 1000),
            'tournament_size': trial.suggest_int('tournament_size', 3, 10)
        }

        # 重复运行3次取平均值，提高稳定性
        n_repeats = 3
        fitness_values = []

        for _ in range(n_repeats):
            from genetic_algorithm import GeneticAlgorithmTSP
            ga = GeneticAlgorithmTSP(self.distance_matrix, **params)
            _, fitness, _ = ga.run()
            fitness_values.append(fitness)

        avg_fitness = np.mean(fitness_values)

        # 记录到历史
        self.optimization_history.append({
            'params': params.copy(),
            'fitness': avg_fitness
        })

        return avg_fitness

    def run_optuna_optimization(self, n_trials=100):
        """使用Optuna进行贝叶斯优化"""
        print("=" * 70)
        print("使用Optuna进行贝叶斯优化调参")
        print("=" * 70)

        # 创建Optuna研究
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )

        # 运行优化
        study.optimize(self.objective_function, n_trials=n_trials, n_jobs=1)

        # 输出最佳参数
        print("\n最佳参数:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        print(f"最佳适应度: {study.best_value:.2f}")

        # 使用最佳参数运行最终验证
        self.best_params = study.best_params
        best_result = self.final_validation(self.best_params)

        return study, best_result

    def visualize_3d_parameter_space(self, study):
        """3D可视化参数空间"""
        print("\n" + "=" * 70)
        print("参数空间3D可视化")
        print("=" * 70)

        # 提取试验数据
        trials = study.trials

        # 准备3D绘图数据
        fig = plt.figure(figsize=(18, 12))

        # 1. 3D参数空间图：种群大小 vs 交叉率 vs 适应度
        ax1 = fig.add_subplot(231, projection='3d')

        pop_sizes = [t.params['population_size'] for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        cross_rates = [t.params['crossover_rate'] for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        fitnesses = [t.value for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

        scatter1 = ax1.scatter(pop_sizes, cross_rates, fitnesses,
                               c=fitnesses, cmap='viridis', s=50, alpha=0.7)
        ax1.set_xlabel('种群大小')
        ax1.set_ylabel('交叉率')
        ax1.set_zlabel('适应度')
        ax1.set_title('种群大小 vs 交叉率 vs 适应度')
        plt.colorbar(scatter1, ax=ax1, label='适应度')

        # 2. 3D参数空间图：交叉率 vs 变异率 vs 适应度
        ax2 = fig.add_subplot(232, projection='3d')

        mutation_rates = [t.params['mutation_rate'] for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

        scatter2 = ax2.scatter(cross_rates, mutation_rates, fitnesses,
                               c=fitnesses, cmap='plasma', s=50, alpha=0.7)
        ax2.set_xlabel('交叉率')
        ax2.set_ylabel('变异率')
        ax2.set_zlabel('适应度')
        ax2.set_title('交叉率 vs 变异率 vs 适应度')
        plt.colorbar(scatter2, ax=ax2, label='适应度')

        # 3. 3D参数空间图：种群大小 vs 变异率 vs 适应度
        ax3 = fig.add_subplot(233, projection='3d')

        scatter3 = ax3.scatter(pop_sizes, mutation_rates, fitnesses,
                               c=fitnesses, cmap='coolwarm', s=50, alpha=0.7)
        ax3.set_xlabel('种群大小')
        ax3.set_ylabel('变异率')
        ax3.set_zlabel('适应度')
        ax3.set_title('种群大小 vs 变异率 vs 适应度')
        plt.colorbar(scatter3, ax=ax3, label='适应度')

        # 4. 参数重要性分析
        ax4 = fig.add_subplot(234)
        plot_param_importances(study).figure_.savefig('temp_param_importance.png')
        param_importance_img = plt.imread('temp_param_importance.png')
        ax4.imshow(param_importance_img)
        ax4.axis('off')
        ax4.set_title('参数重要性分析')

        # 5. 优化历史
        ax5 = fig.add_subplot(235)
        plot_optimization_history(study).figure_.savefig('temp_optimization_history.png')
        history_img = plt.imread('temp_optimization_history.png')
        ax5.imshow(history_img)
        ax5.axis('off')
        ax5.set_title('优化历史')

        # 6. 切片图
        ax6 = fig.add_subplot(236)
        plot_slice(study).figure_.savefig('temp_slice_plot.png')
        slice_img = plt.imread('temp_slice_plot.png')
        ax6.imshow(slice_img)
        ax6.axis('off')
        ax6.set_title('参数切片分析')

        plt.tight_layout()
        plt.show()

        # 清理临时文件
        if os.path.exists('temp_param_importance.png'):
            os.remove('temp_param_importance.png')
        if os.path.exists('temp_optimization_history.png'):
            os.remove('temp_optimization_history.png')
        if os.path.exists('temp_slice_plot.png'):
            os.remove('temp_slice_plot.png')

    def analyze_parameter_interactions_3d(self, study):
        """详细分析参数交互作用的3D可视化"""
        print("\n" + "=" * 70)
        print("参数交互作用详细分析")
        print("=" * 70)

        trials = study.trials
        completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

        # 创建更详细的3D可视化
        fig = plt.figure(figsize=(20, 15))

        # 所有可能的参数组合
        param_pairs = [
            ('population_size', 'crossover_rate', '适应度'),
            ('population_size', 'mutation_rate', '适应度'),
            ('crossover_rate', 'mutation_rate', '适应度'),
            ('population_size', 'elitism_rate', '适应度'),
            ('crossover_rate', 'elitism_rate', '适应度'),
            ('mutation_rate', 'elitism_rate', '适应度'),
        ]

        for i, (param_x, param_y, param_z) in enumerate(param_pairs, 1):
            ax = fig.add_subplot(2, 3, i, projection='3d')

            # 提取数据
            x_data = [t.params.get(param_x, 0) for t in completed_trials]
            y_data = [t.params.get(param_y, 0) for t in completed_trials]

            if param_z == '适应度':
                z_data = [t.value for t in completed_trials]
                color_data = z_data
                cmap = 'viridis_r'  # 反向色图，红色表示好（值小）
            else:
                z_data = [t.params.get(param_z, 0) for t in completed_trials]
                color_data = [t.value for t in completed_trials]
                cmap = 'viridis_r'

            # 创建散点图
            scatter = ax.scatter(x_data, y_data, z_data,
                                 c=color_data, cmap=cmap, s=30, alpha=0.7)

            ax.set_xlabel(param_x)
            ax.set_ylabel(param_y)
            ax.set_zlabel(param_z)
            ax.set_title(f'{param_x} vs {param_y} vs {param_z}')

            plt.colorbar(scatter, ax=ax, label='适应度' if param_z == '适应度' else param_z)

            # 添加最佳点标记
            best_trial = study.best_trial
            if param_x in best_trial.params and param_y in best_trial.params:
                best_x = best_trial.params[param_x]
                best_y = best_trial.params[param_y]
                if param_z == '适应度':
                    best_z = best_trial.value
                elif param_z in best_trial.params:
                    best_z = best_trial.params[param_z]
                else:
                    best_z = 0

                ax.scatter([best_x], [best_y], [best_z],
                           c='red', s=200, marker='*', label='最优解')

        plt.tight_layout()
        plt.show()

        # 创建2D等高线图显示参数交互
        self.create_contour_plots(study)

    def create_contour_plots(self, study):
        """创建等高线图显示参数交互"""
        trials = study.trials
        completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

        # 转换为DataFrame便于分析
        data = []
        for trial in completed_trials:
            row = trial.params.copy()
            row['fitness'] = trial.value
            data.append(row)

        df = pd.DataFrame(data)

        # 创建交互图网格
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        param_combinations = [
            ('population_size', 'crossover_rate'),
            ('population_size', 'mutation_rate'),
            ('crossover_rate', 'mutation_rate'),
            ('population_size', 'elitism_rate'),
            ('crossover_rate', 'elitism_rate'),
            ('mutation_rate', 'elitism_rate'),
        ]

        for idx, (param_x, param_y) in enumerate(param_combinations):
            ax = axes[idx]

            if param_x in df.columns and param_y in df.columns:
                # 创建网格数据
                x_vals = df[param_x].values
                y_vals = df[param_y].values
                z_vals = df['fitness'].values

                # 创建网格
                xi = np.linspace(min(x_vals), max(x_vals), 50)
                yi = np.linspace(min(y_vals), max(y_vals), 50)
                xi, yi = np.meshgrid(xi, yi)

                # 插值
                zi = griddata((x_vals, y_vals), z_vals, (xi, yi), method='cubic')

                # 绘制等高线
                contour = ax.contourf(xi, yi, zi, levels=20, cmap='viridis_r', alpha=0.8)
                ax.scatter(x_vals, y_vals, c=z_vals, cmap='viridis_r',
                           s=30, edgecolors='black', linewidth=0.5)

                # 标记最佳点
                best_trial = study.best_trial
                if param_x in best_trial.params and param_y in best_trial.params:
                    best_x = best_trial.params[param_x]
                    best_y = best_trial.params[param_y]
                    ax.scatter(best_x, best_y, c='red', s=200,
                               marker='*', label='最优解', edgecolors='white')

                ax.set_xlabel(param_x)
                ax.set_ylabel(param_y)
                ax.set_title(f'{param_x} vs {param_y} 交互作用')
                ax.legend()
                ax.grid(True, alpha=0.3)

                plt.colorbar(contour, ax=ax, label='适应度')

        plt.tight_layout()
        plt.show()

    def final_validation(self, best_params, n_validations=10):
        """最终验证实验"""
        print("\n" + "=" * 70)
        print("最终验证实验")
        print("=" * 70)

        validation_results = []
        best_solutions = []
        convergence_histories = []

        for i in range(n_validations):
            print(f"\n验证运行 {i + 1}/{n_validations}")

            start_time = time.time()
            from genetic_algorithm import GeneticAlgorithmTSP
            ga = GeneticAlgorithmTSP(self.distance_matrix, **best_params)
            solution, fitness, history = ga.run()
            elapsed_time = time.time() - start_time

            validation_results.append({
                'run': i + 1,
                'fitness': fitness,
                'time': elapsed_time,
                'solution': solution
            })

            best_solutions.append(solution)
            convergence_histories.append(history)

            print(f"  适应度: {fitness:.2f}, 时间: {elapsed_time:.2f}s")

            # 更新全局最优
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = solution

        # 统计分析
        fitness_values = [r['fitness'] for r in validation_results]
        avg_fitness = np.mean(fitness_values)
        std_fitness = np.std(fitness_values)
        min_fitness = min(fitness_values)
        max_fitness = max(fitness_values)

        print(f"\n验证结果统计:")
        print(f"  平均适应度: {avg_fitness:.2f}")
        print(f"  标准差: {std_fitness:.2f}")
        print(f"  最小值: {min_fitness:.2f}")
        print(f"  最大值: {max_fitness:.2f}")
        print(f"  变异系数: {(std_fitness / avg_fitness * 100):.2f}%")

        # 输出最优解详细信息
        self.output_best_solution_details()

        return {
            'best_params': best_params,
            'validation_results': validation_results,
            'stats': {
                'mean': avg_fitness,
                'std': std_fitness,
                'min': min_fitness,
                'max': max_fitness,
                'cv': std_fitness / avg_fitness * 100
            },
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'convergence_histories': convergence_histories
        }

    def output_best_solution_details(self):
        """输出最优解的详细信息"""
        if self.best_solution is not None:
            print("\n" + "=" * 70)
            print("最优解详细信息")
            print("=" * 70)

            print(f"最优路径长度: {self.best_fitness:.2f}")
            print(f"最优路径城市顺序:")

            # 格式化输出路径
            path_str = " -> ".join([str(i + 1) for i in self.best_solution[:10]])
            if len(self.best_solution) > 10:
                path_str += f" -> ... -> {self.best_solution[-1] + 1}"
            else:
                path_str += f" -> {self.best_solution[0] + 1}"  # 回到起点

            print(f"  路径: {path_str}")

            # 计算路径详细信息
            print("\n路径分段距离:")
            total_distance = 0
            for i in range(len(self.best_solution)):
                city_from = self.best_solution[i]
                city_to = self.best_solution[(i + 1) % len(self.best_solution)]
                distance = self.distance_matrix[city_from][city_to]
                total_distance += distance
                if i < 5 or i >= len(self.best_solution) - 5:  # 显示前后5段
                    print(f"  城市 {city_from + 1} -> 城市 {city_to + 1}: {distance:.2f}")

            if len(self.best_solution) > 10:
                print(f"  ... (省略中间 {len(self.best_solution) - 10} 段) ...")

    def visualize_final_results(self, validation_results, convergence_histories):
        """可视化最终结果"""
        print("\n" + "=" * 70)
        print("结果可视化")
        print("=" * 70)

        fig = plt.figure(figsize=(20, 12))

        # 1. 验证结果分布
        ax1 = fig.add_subplot(231)
        fitness_values = [r['fitness'] for r in validation_results]

        ax1.hist(fitness_values, bins=15, edgecolor='black', alpha=0.7,
                 color='skyblue', density=True)

        # 添加正态分布曲线
        mu, std = norm.fit(fitness_values)
        xmin, xmax = ax1.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax1.plot(x, p, 'r--', linewidth=2, label=f'正态分布拟合\nμ={mu:.2f}, σ={std:.2f}')

        ax1.axvline(np.mean(fitness_values), color='red', linestyle='-',
                    linewidth=2, label=f'平均值: {np.mean(fitness_values):.2f}')
        ax1.axvline(np.min(fitness_values), color='green', linestyle='--',
                    linewidth=2, label=f'最小值: {np.min(fitness_values):.2f}')

        ax1.set_xlabel('路径长度', fontsize=12)
        ax1.set_ylabel('概率密度', fontsize=12)
        ax1.set_title('验证结果分布', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 收敛曲线（前5次运行）
        ax2 = fig.add_subplot(232)
        for i, history in enumerate(convergence_histories[:5]):
            ax2.plot(history, label=f'运行 {i + 1}', alpha=0.7)

        ax2.set_xlabel('迭代次数', fontsize=12)
        ax2.set_ylabel('路径长度', fontsize=12)
        ax2.set_title('收敛曲线', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 运行时间分布
        ax3 = fig.add_subplot(233)
        times = [r['time'] for r in validation_results]
        ax3.boxplot(times, patch_artist=True,
                    boxprops=dict(facecolor='lightgreen'))
        ax3.set_ylabel('运行时间 (秒)', fontsize=12)
        ax3.set_title('运行时间分布', fontsize=14)
        ax3.grid(True, alpha=0.3)

        # 4. 最优路径可视化
        ax4 = fig.add_subplot(234)
        if self.best_solution is not None and hasattr(self, 'cities'):
            x_coords = [city[0] for city in self.cities]
            y_coords = [city[1] for city in self.cities]

            # 绘制城市点
            ax4.scatter(x_coords, y_coords, c='red', s=50, alpha=0.8)

            # 绘制最优路径
            path_x = [x_coords[i] for i in self.best_solution] + [x_coords[self.best_solution[0]]]
            path_y = [y_coords[i] for i in self.best_solution] + [y_coords[self.best_solution[0]]]
            ax4.plot(path_x, path_y, 'b-', alpha=0.6, linewidth=1.5)
            ax4.plot(path_x, path_y, 'ro', markersize=8)

            # 标记起点
            ax4.scatter([x_coords[self.best_solution[0]]], [y_coords[self.best_solution[0]]],
                        c='green', s=200, marker='*', label='起点', edgecolors='black')

            ax4.set_xlabel('X坐标', fontsize=12)
            ax4.set_ylabel('Y坐标', fontsize=12)
            ax4.set_title(f'最优路径 (长度: {self.best_fitness:.2f})', fontsize=14)
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. 参数敏感性雷达图
        ax5 = fig.add_subplot(235, polar=True)
        if self.best_params:
            # 简化显示，最多6个参数
            params_names = list(self.best_params.keys())[:6]
            params_values = list(self.best_params.values())[:6]

            # 归一化值
            normalized_values = [(v - min(params_values)) / (max(params_values) - min(params_values))
                                 if max(params_values) > min(params_values) else 0.5
                                 for v in params_values]

            # 闭合雷达图
            angles = np.linspace(0, 2 * np.pi, len(params_names), endpoint=False).tolist()
            angles += angles[:1]  # 闭合
            normalized_values += normalized_values[:1]

            ax5.plot(angles, normalized_values, 'o-', linewidth=2)
            ax5.fill(angles, normalized_values, alpha=0.25)
            ax5.set_thetagrids(np.degrees(angles[:-1]), params_names)
            ax5.set_title('最优参数配置', fontsize=14, y=1.1)

        # 6. 性能对比（统计摘要）
        ax6 = fig.add_subplot(236)
        ax6.text(0.5, 0.5, f'最优解统计:\n\n'
                           f'平均适应度: {np.mean(fitness_values):.2f}\n'
                           f'标准差: {np.std(fitness_values):.2f}\n'
                           f'最优值: {np.min(fitness_values):.2f}\n'
                           f'变异系数: {(np.std(fitness_values) / np.mean(fitness_values) * 100):.2f}%\n'
                           f'运行次数: {len(validation_results)}',
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax6.transAxes,
                 fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax6.set_title('性能统计摘要', fontsize=14)
        ax6.axis('off')

        plt.tight_layout()
        plt.show()