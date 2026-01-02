import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tsp_loader import load_tsp_file, calculate_distance_matrix
from genetic_algorithm import GeneticAlgorithmTSP
import time
import random

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负


def visualize_tsp_solution(cities, solution, title="TSP Solution"):
    """可视化TSP解"""
    plt.figure(figsize=(10, 8))

    # 提取坐标
    x_coords = [city[0] for city in cities]
    y_coords = [city[1] for city in cities]

    # 绘制城市点
    plt.scatter(x_coords, y_coords, c='red', s=50, alpha=0.8)

    # 绘制路径
    path_x = [x_coords[i] for i in solution] + [x_coords[solution[0]]]
    path_y = [y_coords[i] for i in solution] + [y_coords[solution[0]]]
    plt.plot(path_x, path_y, 'b-', alpha=0.6, linewidth=1.5)
    plt.plot(path_x, path_y, 'ro', markersize=8)

    # 标记城市编号
    for i, (x, y) in enumerate(cities):
        plt.text(x, y, str(i + 1), fontsize=8, ha='center', va='center')

    plt.title(title, fontsize=16)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_parameter_effects(distance_matrix, base_params):
    """分析参数对算法性能的影响"""

    results = {}

    # 分析种群规模影响
    population_sizes = [50, 100, 200, 300]
    pop_results = []

    print("分析种群规模影响...")
    for size in population_sizes:
        params = base_params.copy()
        params['population_size'] = size

        ga = GeneticAlgorithmTSP(distance_matrix, **params)
        start_time = time.time()
        _, fitness, _ = ga.run()
        elapsed_time = time.time() - start_time

        pop_results.append({
            'size': size,
            'fitness': fitness,
            'time': elapsed_time
        })
        print(f"  种群规模 {size}: 适应度={fitness:.2f}, 时间={elapsed_time:.2f}s")

    results['population'] = pop_results

    # 分析交叉概率影响
    crossover_rates = [0.6, 0.7, 0.8, 0.9]
    cross_results = []

    print("\n分析交叉概率影响...")
    for rate in crossover_rates:
        params = base_params.copy()
        params['crossover_rate'] = rate

        ga = GeneticAlgorithmTSP(distance_matrix, **params)
        _, fitness, _ = ga.run()
        cross_results.append({
            'rate': rate,
            'fitness': fitness
        })
        print(f"  交叉概率 {rate}: 适应度={fitness:.2f}")

    results['crossover'] = cross_results

    # 分析变异概率影响
    mutation_rates = [0.1, 0.2, 0.3, 0.4]
    mutation_results = []

    print("\n分析变异概率影响...")
    for rate in mutation_rates:
        params = base_params.copy()
        params['mutation_rate'] = rate

        ga = GeneticAlgorithmTSP(distance_matrix, **params)
        _, fitness, _ = ga.run()
        mutation_results.append({
            'rate': rate,
            'fitness': fitness
        })
        print(f"  变异概率 {rate}: 适应度={fitness:.2f}")

    results['mutation'] = mutation_results

    return results


def run_genetic_algorithm_optimization(distance_matrix, params_list):
    """运行多个参数配置的遗传算法"""
    results = []

    for i, params in enumerate(params_list, 1):
        print(f"\n运行配置 {i}/{len(params_list)}:")
        print(f"  参数: {params}")

        ga = GeneticAlgorithmTSP(distance_matrix, **params)
        start_time = time.time()
        best_solution, best_fitness, fitness_history = ga.run()
        elapsed_time = time.time() - start_time

        results.append({
            'params': params,
            'fitness': best_fitness,
            'time': elapsed_time,
            'solution': best_solution,
            'history': fitness_history
        })

        print(f"  结果: 适应度={best_fitness:.2f}, 时间={elapsed_time:.2f}s")

    return results


def manual_parameter_tuning(distance_matrix):
    """手动参数调优"""
    # 定义多个参数配置
    param_configs = [
        # 配置1: 基础配置
        {
            'population_size': 100,
            'generations': 500,
            'crossover_rate': 0.8,
            'mutation_rate': 0.2,
            'elitism_rate': 0.1
        },
        # 配置2: 大种群
        {
            'population_size': 200,
            'generations': 300,
            'crossover_rate': 0.85,
            'mutation_rate': 0.15,
            'elitism_rate': 0.15
        },
        # 配置3: 高变异
        {
            'population_size': 150,
            'generations': 400,
            'crossover_rate': 0.75,
            'mutation_rate': 0.3,
            'elitism_rate': 0.1
        },
        # 配置4: 长期进化
        {
            'population_size': 80,
            'generations': 800,
            'crossover_rate': 0.9,
            'mutation_rate': 0.1,
            'elitism_rate': 0.05
        }
    ]

    results = run_genetic_algorithm_optimization(distance_matrix, param_configs)

    # 找出最佳配置
    best_result = min(results, key=lambda x: x['fitness'])
    print(f"\n{'=' * 60}")
    print(f"最佳配置: 适应度={best_result['fitness']:.2f}")
    print(f"参数: {best_result['params']}")
    print(f"运行时间: {best_result['time']:.2f}s")
    print('=' * 60)

    return results, best_result


def visualize_results_comparison(results):
    """可视化不同配置的结果对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 适应度对比
    config_names = [f"配置{i + 1}" for i in range(len(results))]
    fitness_values = [r['fitness'] for r in results]

    ax1 = axes[0, 0]
    bars = ax1.bar(config_names, fitness_values, color='skyblue')
    ax1.set_title('不同配置的适应度对比', fontsize=14)
    ax1.set_ylabel('适应度（路径长度）', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)

    # 在柱子上显示数值
    for bar, val in zip(bars, fitness_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{val:.1f}', ha='center', va='bottom')

    # 2. 运行时间对比
    time_values = [r['time'] for r in results]

    ax2 = axes[0, 1]
    bars = ax2.bar(config_names, time_values, color='lightcoral')
    ax2.set_title('不同配置的运行时间对比', fontsize=14)
    ax2.set_ylabel('时间（秒）', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)

    for bar, val in zip(bars, time_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{val:.2f}s', ha='center', va='bottom')

    # 3. 收敛曲线对比
    ax3 = axes[1, 0]
    for i, result in enumerate(results):
        history = result['history']
        ax3.plot(history, label=f'配置{i + 1}', linewidth=2)

    ax3.set_title('收敛曲线对比', fontsize=14)
    ax3.set_xlabel('迭代次数', fontsize=12)
    ax3.set_ylabel('最佳适应度', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 参数雷达图
    ax4 = axes[1, 1]

    # 提取参数并归一化
    params_to_plot = ['population_size', 'crossover_rate', 'mutation_rate']
    categories = ['种群规模', '交叉概率', '变异概率']

    for i, result in enumerate(results):
        params = result['params']
        values = []

        # 归一化值
        pop_sizes = [r['params']['population_size'] for r in results]
        cross_rates = [r['params']['crossover_rate'] for r in results]
        mut_rates = [r['params']['mutation_rate'] for r in results]

        values.append((params['population_size'] - min(pop_sizes)) / (max(pop_sizes) - min(pop_sizes)))
        values.append((params['crossover_rate'] - min(cross_rates)) / (max(cross_rates) - min(cross_rates)))
        values.append((params['mutation_rate'] - min(mut_rates)) / (max(mut_rates) - min(mut_rates)))

        # 闭合雷达图
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        ax4.plot(angles, values, 'o-', linewidth=2, label=f'配置{i + 1}')
        ax4.fill(angles, values, alpha=0.25)

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_title('参数配置对比雷达图', fontsize=14)
    ax4.legend(loc='upper right')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    random.seed(42)

    # 加载TSP数据
    print("=== 加载TSP50数据 ===")
    cities = load_tsp_file('TSP50.txt')
    distance_matrix = calculate_distance_matrix(cities)

    print(f"城市数量: {len(cities)}")
    print(f"距离矩阵维度: {len(distance_matrix)}x{len(distance_matrix[0])}")

    # 实验1: 手动参数调优
    print("\n" + "=" * 60)
    print("实验1: 手动参数调优")
    print("=" * 60)

    all_results, best_result = manual_parameter_tuning(distance_matrix)

    # 可视化结果对比
    visualize_results_comparison(all_results)

    # 实验2: 使用最佳参数运行详细实验
    print("\n" + "=" * 60)
    print("实验2: 使用最佳参数进行详细实验")
    print("=" * 60)

    best_params = best_result['params']
    print(f"使用最佳参数: {best_params}")

    ga = GeneticAlgorithmTSP(distance_matrix, **best_params)
    best_solution, best_fitness, fitness_history = ga.run()

    print(f"\n最终结果:")
    print(f"最佳路径长度: {best_fitness:.2f}")
    print(f"最佳路径前10个城市: {best_solution[:10]}...")
    print(f"最佳路径完整顺序: {best_solution}")

    # 可视化最佳路径
    visualize_tsp_solution(cities, best_solution,
                           f"TSP50最佳路径 (长度: {best_fitness:.2f})")

    # 实验3: 参数影响分析
    print("\n" + "=" * 60)
    print("实验3: 参数影响分析")
    print("=" * 60)

    # 使用中等参数进行分析
    base_params = {
        'population_size': 100,
        'generations': 300,
        'crossover_rate': 0.8,
        'mutation_rate': 0.2,
        'elitism_rate': 0.1
    }

    parameter_results = analyze_parameter_effects(distance_matrix, base_params)

    # 可视化参数影响
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 种群规模 vs 适应度
    pop_data = parameter_results['population']
    ax1 = axes[0, 0]
    ax1.plot([d['size'] for d in pop_data], [d['fitness'] for d in pop_data],
             'o-', linewidth=2, markersize=8)
    ax1.set_title('种群规模 vs 适应度', fontsize=14)
    ax1.set_xlabel('种群规模', fontsize=12)
    ax1.set_ylabel('最佳适应度', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 2. 种群规模 vs 时间
    ax2 = axes[0, 1]
    ax2.plot([d['size'] for d in pop_data], [d['time'] for d in pop_data],
             's-', color='orange', linewidth=2, markersize=8)
    ax2.set_title('种群规模 vs 时间', fontsize=14)
    ax2.set_xlabel('种群规模', fontsize=12)
    ax2.set_ylabel('运行时间 (秒)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 3. 交叉概率影响
    cross_data = parameter_results['crossover']
    ax3 = axes[0, 2]
    ax3.plot([d['rate'] for d in cross_data], [d['fitness'] for d in cross_data],
             '^-', color='green', linewidth=2, markersize=8)
    ax3.set_title('交叉概率 vs 适应度', fontsize=14)
    ax3.set_xlabel('交叉概率', fontsize=12)
    ax3.set_ylabel('最佳适应度', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # 4. 变异概率影响
    mut_data = parameter_results['mutation']
    ax4 = axes[1, 0]
    ax4.plot([d['rate'] for d in mut_data], [d['fitness'] for d in mut_data],
             'd-', color='red', linewidth=2, markersize=8)
    ax4.set_title('变异概率 vs 适应度', fontsize=14)
    ax4.set_xlabel('变异概率', fontsize=12)
    ax4.set_ylabel('最佳适应度', fontsize=12)
    ax4.grid(True, alpha=0.3)

    # 5. 收敛曲线
    ax5 = axes[1, 1]
    ax5.plot(fitness_history, linewidth=2)
    ax5.set_title('最佳配置收敛曲线', fontsize=14)
    ax5.set_xlabel('迭代次数', fontsize=12)
    ax5.set_ylabel('最佳适应度', fontsize=12)
    ax5.grid(True, alpha=0.3)

    # 6. 适应度分布箱线图
    ax6 = axes[1, 2]

    # 模拟多次运行的适应度分布
    np.random.seed(42)
    sample_fitness = []
    for _ in range(5):
        ga_temp = GeneticAlgorithmTSP(distance_matrix, **base_params)
        _, fitness, _ = ga_temp.run()
        sample_fitness.append(fitness)

    ax6.boxplot(sample_fitness, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red'))
    ax6.set_title('多次运行适应度分布', fontsize=14)
    ax6.set_ylabel('适应度', fontsize=12)
    ax6.set_xticklabels(['运行1-5'])
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 实验4: 路径质量分析
    print("\n" + "=" * 60)
    print("实验4: 路径质量分析")
    print("=" * 60)

    # 计算相邻城市距离
    distances = []
    for i in range(len(best_solution)):
        from_city = best_solution[i]
        to_city = best_solution[(i + 1) % len(best_solution)]
        distances.append(distance_matrix[from_city][to_city])

    print(f"路径分段距离统计:")
    print(f"  最短分段: {min(distances):.2f}")
    print(f"  最长分段: {max(distances):.2f}")
    print(f"  平均分段: {np.mean(distances):.2f}")
    print(f"  标准差: {np.std(distances):.2f}")

    # 可视化距离分布
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(np.mean(distances), color='red', linestyle='--',
                linewidth=2, label=f'平均值: {np.mean(distances):.2f}')
    plt.title('路径分段距离分布', fontsize=16)
    plt.xlabel('距离', fontsize=14)
    plt.ylabel('频数', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()