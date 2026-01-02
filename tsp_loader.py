import math


def load_tsp_file(filename):
    """读取TSP文件，提取城市坐标（文件编码为GB2312）"""
    cities = []
    with open(filename, 'r', encoding='GB2312') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行

            # 分割并解析数字部分（城市编号、x、y）
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():  # 确保是有效城市行
                city_id = int(parts[0]) - 1  # 转为0-based索引
                x, y = float(parts[1]), float(parts[2])

                # 确保列表长度足够，填充可能的空缺（实际数据中通常连续）
                if len(cities) <= city_id:
                    cities.extend([None] * (city_id - len(cities) + 1))
                cities[city_id] = (x, y)

    print(f"成功加载 {len(cities)} 个城市")
    return cities


def calculate_distance_matrix(cities):
    """计算城市间的欧氏距离矩阵"""
    n = len(cities)
    distance_matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        x1, y1 = cities[i]
        for j in range(i + 1, n):
            x2, y2 = cities[j]
            distance = math.hypot(x2 - x1, y2 - y1)  # 简化距离计算
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    return distance_matrix


