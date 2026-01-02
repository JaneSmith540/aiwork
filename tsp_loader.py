# tsp_loader.py
import math
import chardet


def detect_file_encoding(filename):
    """检测文件编码"""
    with open(filename, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        print(f"检测到文件编码: {encoding} ")
        return encoding


def load_tsp_file(filename):
    """
    读取TSP文件，提取城市坐标
    支持自动编码检测
    """
    # 检测并读取文件
    encoding = detect_file_encoding(filename)

    cities = []
    with open(filename, 'r', encoding=encoding, errors='ignore') as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:  # 跳过空行
                continue

            # 尝试解析数字，忽略非数字字符
            parts = []
            for token in line.split():
                # 只保留可以转换为数字的部分
                clean_token = ''
                for char in token:
                    if char.isdigit() or char == '.' or (char == '-' and not clean_token):
                        clean_token += char
                    elif clean_token:  # 遇到非数字字符且已有数字，结束
                        break

                if clean_token:
                    try:
                        num = float(clean_token)
                        parts.append(num)
                    except ValueError:
                        continue

            # 至少需要3个数字：编号、X、Y
            if len(parts) >= 3:
                city_id = int(parts[0]) - 1  # 转换为0-based索引
                x = parts[1]
                y = parts[2]

                # 确保数组足够大
                if len(cities) <= city_id:
                    cities.extend([None] * (city_id - len(cities) + 1))
                cities[city_id] = (x, y)

    # 移除可能的None值
    cities = [city for city in cities if city is not None]
    print(f"成功加载 {len(cities)} 个城市")
    return cities


def calculate_distance_matrix(cities):
    """
    计算城市间的距离矩阵
    """
    n = len(cities)
    distance_matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        x1, y1 = cities[i]
        for j in range(i + 1, n):
            x2, y2 = cities[j]
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    return distance_matrix


def debug_tsp_file(filename):
    """调试TSP文件内容"""
    encoding = detect_file_encoding(filename)
    print(f"\n使用 {encoding} 编码读取文件内容:")
    print("-" * 50)

    with open(filename, 'r', encoding=encoding, errors='ignore') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:15]):  # 只显示前15行
            print(f"行{i + 1}: {repr(line)}")

    print("-" * 50)

    # 加载城市
    cities = load_tsp_file(filename)

    if cities:
        print("\n加载的城市坐标:")
        print(f"总城市数: {len(cities)}")
        print("\n前5个城市:")
        for i, (x, y) in enumerate(cities[:5]):
            print(f"城市{i + 1}: ({x}, {y})")

        if len(cities) > 5:
            print(f"... 和 {len(cities) - 5} 个更多城市")

        # 计算并显示距离矩阵示例
        distance_matrix = calculate_distance_matrix(cities)
        print(f"\n距离矩阵示例 (3x3):")
        for i in range(min(3, len(cities))):
            for j in range(min(3, len(cities))):
                print(f"{distance_matrix[i][j]:8.2f}", end=' ')
            print()

    return cities


# 测试函数
if __name__ == "__main__":
    filename = "TSP50.txt"
    print(f"开始处理文件: {filename}")

    try:
        cities = debug_tsp_file(filename)

        if cities:
            # 验证距离计算
            distance_matrix = calculate_distance_matrix(cities)
            print(f"\n距离矩阵验证:")
            print(f"矩阵大小: {len(distance_matrix)}x{len(distance_matrix[0])}")
            print(f"城市1到城市2的距离: {distance_matrix[0][1]:.2f}")
            print(f"城市2到城市3的距离: {distance_matrix[1][2]:.2f}")
            print(f"对称性验证: {distance_matrix[0][1] == distance_matrix[1][0]}")
    except Exception as e:
        print(f"处理文件时出错: {e}")
        import traceback

        traceback.print_exc()