import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode


# 模糊综合评价函数
def fuzzy_comprehensive_evaluation(fuzzy_matrix, weights):
    """
    模糊综合评价计算。
    :param fuzzy_matrix: numpy array, 模糊评价矩阵（形状为 m x n，m 为评分等级数，n 为评价指标数）
    :param weights: numpy array, 权重向量（形状为 n，权重之和为 1）
    :return: numpy array, 模糊综合结果（形状为 m）
    """
    # 确保权重向量的和为 1
    weights = weights / np.sum(weights)

    # 模糊综合运算：使用矩阵乘法进行加权平均
    comprehensive_result = np.dot(fuzzy_matrix, weights)

    return comprehensive_result


# 主函数
def main():
    # 1. 确定评价指标体系
    evaluation_criteria = ["U1", "U2", "U3", "U4"]  # 示例指标体系
    print("评价指标体系：", evaluation_criteria)

    # 2. 建立模糊评价矩阵（示例）
    # 假设共有 5 个评分等级（例如：非常差、差、一般、好、非常好）
    # 每个指标的评分由专家打分法得到，或是使用模糊隶属函数确定模糊隶属度
    fuzzy_matrix = np.array([
        [0.1, 0.1, 0.3, 0.4, 0.1],  # 指标 U1
        [0.2, 0.2, 0.2, 0.1, 0.3],  # 指标 U2
        [0.1, 0.2, 0.3, 0.2, 0.2],  # 指标 U3
        [0.2, 0.0, 0.1, 0.3, 0.4],  # 指标 U4
    ]).T

    print("\n模糊评价矩阵：\n", fuzzy_matrix)

    # 3. 确定权重向量
    # 假设权重通过专家打分法或层次分析法得到
    weights = np.array([0.2, 0.3, 0.4, 0.1])
    print("\n权重向量：", weights)

    # 4. 模糊综合评价
    results = fuzzy_comprehensive_evaluation(fuzzy_matrix, weights)
    print("\n模糊综合评价结果（每个评分等级的隶属度）：\n", results)

    # 5. 最终结果分析
    # 计算出最终得分
    final_score = 0
    for i in range(len(results)):
        final_score += results[i] * (i * 0.1)  # 评分等级 i 得分为 i * 0.1
    final_score = final_score / len(results) / 0.1 * 100
    print("\n最终得分：", final_score.round(2))


# 调用主函数
if __name__ == "__main__":
    main()
