import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 1. 生成或加载数据
    # 假设我们已经有一个数据集，包含五列：'Income', 'Education_Level', 'Work_Experience', 'Spending', 'Loan_Amount'
    # 以下是生成随机数据的例子
    np.random.seed(42)
    n_samples = 1000
    data = pd.DataFrame(
        {
            "Income": np.random.randint(30000, 200000, size=n_samples),
            "Education_Level": np.random.randint(12, 22, size=n_samples),
            "Work_Experience": np.random.randint(0, 30, size=n_samples),
            "Spending": np.random.randint(1000, 10000, size=n_samples),
            "Loan_Amount": np.random.randint(10000, 500000, size=n_samples),
        }
    )

    # 打印数据前几行
    print("数据前几行：")
    print(data.head())

    # 2. 因子分析
    fa = FactorAnalysis(n_components=2, random_state=42)
    fa.fit(data)

    # 3. 查看因子载荷矩阵
    # 因子载荷矩阵表示每个变量与各个因子之间的关系
    factor_loadings = fa.components_.T
    print("\n因子载荷矩阵：")
    print(pd.DataFrame(factor_loadings, index=data.columns, columns=["Factor 1", "Factor 2"]))

    # 4. 可视化因子载荷
    plt.figure(figsize=(10, 6))
    sns.heatmap(pd.DataFrame(factor_loadings, columns=[f"Factor_{i + 1}" for i in range(2)], index=data.columns),
                annot=True, cmap="coolwarm", cbar=True)
    plt.title("Heatmap of Factor Loadings")
    plt.show()

    # 5. 查看因子分析的解释方差比例
    explained_variances = fa.noise_variance_
    print("\n各因子的噪声方差（未解释部分）：")
    print(explained_variances)

    # 6. 完整的结果解释
    # 根据因子载荷矩阵，我们可以分析每个变量对各个因子的贡献。
    print("\n因子解释：")
    factor1 = factor_loadings[:, 0]
    factor2 = factor_loadings[:, 1]

    # 打印每个因子的主要变量
    print("因子1的主要变量（载荷较大）：")
    print(data.columns[factor1 > 0.5])  # 假设载荷大于 0.5 表示重要
    print("因子2的主要变量（载荷较大）：")
    print(data.columns[factor2 > 0.5])


if __name__ == '__main__':
    main()
