# 导入必要的库
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def main():
    # 1. 生成或加载数据
    # 为了演示，我们生成一个虚拟的数据集
    # 使用 scikit-learn 的 make_blobs 生成数据
    np.random.seed(42)
    X, y = make_blobs(n_samples=300, n_features=4, centers=3, cluster_std=2.0, random_state=42)

    # 2. 数据预处理
    # 对特征进行标准化，这是 PCA 的重要步骤
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 使用 PCA 进行降维
    # 创建 PCA 模型，指定降维后的维度为 2（便于可视化）
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 查看每个主成分对应的原始特征权重
    components = pca.components_
    print("\n每个主成分对应的原始特征权重：")
    print(components)

    # 4. 查看 PCA 的结果
    # 打印保留的方差比例
    explained_variance_ratio = pca.explained_variance_ratio_
    print("\nPCA 保留的方差比例：")
    print(explained_variance_ratio)

    # 查看降维后的数据形状
    print("\n降维前的形状：", X_scaled.shape)
    print("降维后的形状：", X_pca.shape)

    # 5. 可视化降维后的数据
    plt.figure(figsize=(10, 6))

    # 绘制降维后的数据，按类别着色
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA reduced data')
    plt.colorbar(label='category')

    # 显示图形
    plt.show()

    # 6. 提取出降维后的二维数据
    X_pca_2d = X_pca[:, :2]
    print("\n降维后的二维数据：")
    print(X_pca_2d)


if __name__ == '__main__':
    main()
