import numpy as np
import math

# 第一步：熵权法确定指标权重

# 原始数据（x1, x2是利益型指标，x3是成本型指标）
data = np.array([
    [100, 300, 500],  # 项目 A
    [200, 250, 400],  # 项目 B
    [150, 350, 450]   # 项目 C
])


# 标准化处理
def standardize_data(data):
    n_projects, n_indicators = data.shape
    standardized_data = np.zeros_like(data, dtype=float)

    for j in range(n_indicators):
        if j in [0, 1]:  # 利益型指标（x1, x2）
            max_value = np.max(data[:, j])
            standardized_data[:, j] = data[:, j] / max_value
        elif j == 2:  # 成本型指标（x3）
            min_value = np.min(data[:, j])
            standardized_data[:, j] = min_value / data[:, j]

    return standardized_data


# 计算熵值
def calculate_entropy(standardized_data):
    n_projects, n_indicators = standardized_data.shape
    entropy = np.zeros(n_indicators)

    for j in range(n_indicators):
        pij = standardized_data[:, j] / np.sum(standardized_data[:, j])
        entropy[j] = - np.sum(pij * np.log(pij)) / np.log(n_projects)

    return entropy


# 计算权重
def calculate_weights(entropy):
    weights = (1 - entropy) / np.sum(1 - entropy)
    return weights


# 第二步：TOPSIS 方法对方案进行评价和排序

# 计算加权标准化决策矩阵
def calculate_weighted_matrix(standardized_data, weights):
    return standardized_data * weights


# 计算正理想解和负理想解
def calculate_ideal_solutions(weighted_matrix):
    positive_ideal = np.max(weighted_matrix, axis=0)
    negative_ideal = np.min(weighted_matrix, axis=0)
    return positive_ideal, negative_ideal


# 计算各方案与正负理想值的距离
def calculate_distances(weighted_matrix, positive_ideal, negative_ideal):
    n_projects, n_indicators = weighted_matrix.shape
    distances = np.zeros((n_projects, 2))

    for i in range(n_projects):
        distances[i, 0] = np.sqrt(np.sum((weighted_matrix[i, :] - positive_ideal) ** 2))  # 距离正理想解
        distances[i, 1] = np.sqrt(np.sum((weighted_matrix[i, :] - negative_ideal) ** 2))  # 距离负理想解

    return distances


# 计算相对接近度
def calculate_closeness_ratios(distances):
    closeness_ratio = distances[:, 1] / (distances[:, 0] + distances[:, 1])
    return closeness_ratio


# 主函数
def main():
    # 1. 标准化处理
    standardized_data = standardize_data(data)
    print("标准化后的数据：")
    print(standardized_data)

    # 2. 计算熵值
    entropy = calculate_entropy(standardized_data)
    print("\n熵值：")
    print(entropy)

    # 3. 计算权重
    weights = calculate_weights(entropy)
    print("\n权重：")
    print(weights)

    # 4. TOPSIS 步骤
    # 4.1 计算加权标准化决策矩阵
    weighted_matrix = calculate_weighted_matrix(standardized_data, weights)
    print("\n加权标准化决策矩阵：")
    print(weighted_matrix)

    # 4.2 确定正理想解和负理想解
    positive_ideal, negative_ideal = calculate_ideal_solutions(weighted_matrix)
    print("\n正理想解：", positive_ideal)
    print("负理想解：", negative_ideal)

    # 4.3 计算各方案与正负理想值的距离
    distances = calculate_distances(weighted_matrix, positive_ideal, negative_ideal)
    print("\n各方案与正负理想值的距离：")
    print("距离正理想解：", distances[:, 0])
    print("距离负理想解：", distances[:, 1])

    # 4.4 计算相对接近度
    closeness_ratio = calculate_closeness_ratios(distances)
    print("\n相对接近度（越接近 1 越好）：")
    print(closeness_ratio)

    # 5. 排序结果
    print("\n排序结果（按相对接近度降序）：")
    sorted_indices = np.argsort(-closeness_ratio)
    print(sorted_indices)


# 执行主函数
if __name__ == "__main__":
    main()
