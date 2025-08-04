import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  sklearn import datasets
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

def main():
    # 1. 加载鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 2. 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 核主成分分析
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
    X_kpca = kpca.fit_transform(X_scaled)

    # 4. 创建DataFrame保存降维结果
    df_kpca = pd.DataFrame(X_kpca, columns=['Principal Component 1', 'Principal Component 2'])
    df_kpca['Target'] = y

    # 5. 绘制降维结果
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b']
    for target, color in zip([0, 1, 2], colors):
        indices = df_kpca['Target'] == target
        plt.scatter(df_kpca.loc[indices, 'Principal Component 1'],
                    df_kpca.loc[indices, 'Principal Component 2'],
                    c=color, alpha=0.8, label=target)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title('Kernel PCA of Iris Dataset')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
