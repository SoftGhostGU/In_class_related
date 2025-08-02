import numpy as np
import matplotlib.pyplot as plt

# 原始学生成绩数据（每行一个学生，每列一门科目）
# 假设数据包含数学、物理、英语、语文四门科目
scores = np.array([
    [90, 85, 80, 95],  # 学生 A
    [85, 90, 85, 85],  # 学生 B
    [80, 80, 90, 80],  # 学生 C
    [90, 85, 90, 90],  # 学生 D
])


# 数据标准化函数
def standardize_data(data):
    """
    对数据进行标准化处理。
    :param data: numpy array, 形状为 n_student x n_subjects
    :return: 标准化后的数据
    """
    max_values = np.max(data, axis=0)  # 每一列的最大值
    standardized_data = data / max_values  # 标准化
    return standardized_data


# 差异序列计算函数
def calculate_difference_series(reference, data):
    """
    计算差异序列。
    :param reference: numpy array, 参考序列（最优学生）
    :param data: numpy array, 待分析的数据
    :return: 差异序列
    """
    difference_series = np.abs(data - reference)  # 差异值的绝对值
    return difference_series


# 灰色关联度计算函数
def calculate_grey_correlation(difference_series, rho=0.5):
    """
    计算灰色关联度。
    :param difference_series: numpy array, 差异序列
    :param rho: float, 分辨系数，默认为 0.5
    :return: numpy array, 灰色关联度
    """
    delta_max = np.max(difference_series, axis=0)  # 最大差异值
    delta_min = np.min(difference_series, axis=0)  # 最小差异值
    grey_correlation = (delta_min + rho * delta_max) / (difference_series + rho * delta_max)
    return grey_correlation


# 主函数
def main():
    # 1. 数据标准化
    standardized_scores = standardize_data(scores)
    print("标准化后的数据：")
    print(standardized_scores)

    # 2. 确定最优学生
    # (方法 2) 选择每列的最大值作为参考序列
    reference_student = np.max(standardized_scores, axis=0)  # 每门科目的最大值
    print("\n参考序列（每列的最大值）：")
    print(reference_student)

    # 3. 计算差异序列
    difference_series = calculate_difference_series(reference_student, standardized_scores)
    print("\n差异序列：")
    print(difference_series)

    # 4. 计算灰色关联度
    grey_correlation = calculate_grey_correlation(difference_series)
    print("\n灰色关联度：")
    print(grey_correlation)

    # 5. 计算灰色关联度的均值（综合评价）
    overall_grey_CORRELATION = np.mean(grey_correlation, axis=1)
    print("\n学生与最优学生的灰色关联度平均值：")
    print(overall_grey_CORRELATION)

    # 6. 可视化结果
    # student_labels = [f"学生 {chr(65 + i)}" for i in range(len(scores))]
    # plt.bar(student_labels, overall_grey_CORRELATION)
    # plt.xlabel("学生")
    # plt.ylabel("灰色关联度均值")
    # plt.title("学生与最优学生的关联度对比")
    # plt.show()


# 调用主函数
if __name__ == "__main__":
    main()
