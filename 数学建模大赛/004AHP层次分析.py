import numpy as np

# 1. 计算判断矩阵的权重向量和一致性比率
def calculate_weights(judgment_matrix):
    """
    计算判断矩阵的权重向量，并进行一致性检验。
    :param judgment_matrix: numpy array, 判断矩阵（正互反矩阵）
    :return: 权重向量、一致性比率 CR
    """
    # (1) 获取矩阵的维度
    n = judgment_matrix.shape[0]

    # (2) 计算最大特征值和对应的特征向量
    eigenvalues, eigenvectors = np.linalg.eig(judgment_matrix)
    max_eigenvalue = np.max(eigenvalues)
    max_eigenvalue_index = np.argmax(eigenvalues)
    eigenvector = eigenvectors[:, max_eigenvalue_index]
    weights = np.abs(eigenvector) / np.sum(np.abs(eigenvector))  # 归一化

    # (3) 一致性检验
    CI = (max_eigenvalue - n) / (n - 1)  # 一致性指标 CI
    RI = {
        1: 0, 2: 0, 3: 0.52, 4: 0.89, 5: 1.12, 6: 1.26, 7: 1.36, 8: 1.41, 9: 1.46, 10: 1.49
    }[n]  # 随机场乱一致性指标 RI
    CR = CI / RI  # 一致性比率 CR

    return weights, CR

# 2. 获取准则权重
def get_criterion_weights(criteria_judgment_matrix):
    """
    计算准则权重。
    :param criteria_judgment_matrix: numpy array, 准则间的判断矩阵
    :return: 权重向量、一致性比率 CR
    """
    weights, cr = calculate_weights(criteria_judgment_matrix)
    return weights, cr

# 3. 获取备选方案权重
def get_alternative_weights(criteria_weights, alternatives_judgment_matrices):
    """
    计算备选方案的权重。
    :param criteria_weights: numpy array, 每个准则的权重
    :param alternatives_judgment_matrices: list of numpy arrays, 每个准则下备选方案的判断矩阵
    :return: 最终备选方案的权重向量
    """
    criterion_count = len(criteria_weights)
    alternative_count = alternatives_judgment_matrices[0].shape[0]

    # (1) 计算每个准则下的权重向量
    weights_per_criterion = []
    for judgment_matrix in alternatives_judgment_matrices:
        weights, _ = calculate_weights(judgment_matrix)
        weights_per_criterion.append(weights)

    # (2) 加权平均得到最终备选方案权重
    final_alternative_weights = np.zeros(alternative_count)
    for i in range(criterion_count):
        final_alternative_weights += criteria_weights[i] * weights_per_criterion[i]

    return final_alternative_weights

# 4. 综合评价
def comprehensive_evaluation(criteria_judgment_matrix, alternatives_judgment_matrices):
    """
    完成 AHP 层次分析法的综合评价。
    :param criteria_judgment_matrix: numpy array, 准则判断矩阵
    :param alternatives_judgment_matrices: list of numpy arrays, 每个准则下备选方案的判断矩阵
    :return: 最终备选方案权重
    """
    # (1) 获取准则权重
    print("### 步骤 1: 获取准则权重")
    criterion_weights, cr = get_criterion_weights(criteria_judgment_matrix)
    print(f"准则权重: {criterion_weights}")
    print(f"一致性比率 CR: {cr:.4f}")

    if cr > 0.1:
        raise ValueError("判断矩阵的一致性比率 CR > 0.1，不满足一致性要求。请重新调整判断矩阵。")

    # (2) 获取备选方案权重
    print("\n### 步骤 2: 获取备选方案权重")
    final_alternative_weights = get_alternative_weights(criterion_weights, alternatives_judgment_matrices)
    print(f"最终备选方案权重: {final_alternative_weights}")

    return final_alternative_weights

# 主函数
def main():
    # 示例数据
    # (1) 准则矩阵
    criteria_judgment_matrix = np.array([
        [1, 3, 5],          # 准则 U1 相对 U2、U3 的重要性
        [1/3, 1, 2],        # 准则 U2 相对 U1、U3 的重要性
        [1/5, 1/2, 1]       # 准则 U3 相对 U1、U2 的重要性
    ])
    print("\n准则判断矩阵：\n", criteria_judgment_matrix)

    # (2) 备选方案矩阵（针对每个准则）
    alternatives_judgment_matrices = [
        np.array([
            [1, 3, 1/5],     # 备选方案 A1、A2、A3 相对重要性（准则 1）
            [1/3, 1, 1/7],
            [5, 7, 1]
        ]),
        np.array([
            [1, 1/3, 1/7],   # 备选方案 A1、A2、A3 相对重要性（准则 2）
            [3, 1, 1/3],
            [7, 3, 1]
        ]),
        np.array([
            [1, 5, 9],       # 备选方案 A1、A2、A3 相对重要性（准则 3）
            [1/5, 1, 3],
            [1/9, 1/3, 1]
        ])
    ]

    # (3) 进行综合评价
    print("\n### AHP 层次分析综合评价")
    final_weights = comprehensive_evaluation(criteria_judgment_matrix, alternatives_judgment_matrices)
    print("\n最终评价结果：", final_weights)

# 调用主函数
if __name__ == "__main__":
    main()
