import numpy as np
import random
import json
from tsp_loader import load_tsp_file, calculate_distance_matrix
from optuna_optimizer import TSPSystematicOptimizer


def main():
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    random.seed(42)

    # 加载TSP数据
    print("=== 加载TSP数据 ===")
    # 切换数据集：TSP50.txt 或 TSP280.txt
    cities = load_tsp_file('TSP280.txt')
    distance_matrix = calculate_distance_matrix(cities)

    print(f"城市数量: {len(cities)}")
    print(f"距离矩阵维度: {len(distance_matrix)}x{len(distance_matrix[0])}")

    # 创建优化器
    optimizer = TSPSystematicOptimizer(distance_matrix, cities)

    # 使用Optuna进行优化
    study, validation_result = optimizer.run_optuna_optimization(n_trials=100)

    # 3D可视化参数空间
    optimizer.visualize_3d_parameter_space(study)

    # 详细分析参数交互作用
    optimizer.analyze_parameter_interactions_3d(study)

    # 可视化最终结果
    optimizer.visualize_final_results(
        validation_result['validation_results'],
        validation_result['convergence_histories']
    )

    # 输出最优解详细信息
    optimizer.output_best_solution_details()

    # 保存最佳结果
    print("\n" + "=" * 70)
    print("优化完成！")
    print("=" * 70)
    print(f"最佳参数配置: {optimizer.best_params}")
    print(f"最佳路径长度: {optimizer.best_fitness:.2f}")

    # 导出结果到文件
    results = {
        'best_params': optimizer.best_params,
        'best_fitness': optimizer.best_fitness,
        'best_solution': optimizer.best_solution.tolist() if hasattr(optimizer.best_solution,
                                                                     'tolist') else optimizer.best_solution,
        'validation_stats': validation_result['stats']
    }

    with open('tsp_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("结果已保存到 tsp_optimization_results.json")


if __name__ == "__main__":
    main()