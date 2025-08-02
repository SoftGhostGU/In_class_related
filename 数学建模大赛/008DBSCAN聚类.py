import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main():
    # 1. 生成模拟数据，包含多个簇和噪声点
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)
    # 添加一些噪声点
    X = np.vstack([X, np.random.uniform(low=-3, high=3, size=(50, 2))])

    # 2. 数据标准化
    X_scaled = StandardScaler().fit_transform(X)

    # 3. 使用DBSCAN进行聚类
    # 初始化DBSCAN模型
    dbscan = DBSCAN(eps=0.3, min_samples=5)

    # 进行聚类
    dbscan.fit(X_scaled)

    # 获取聚类结果
    labels = dbscan.labels_

    # 4. 绘制聚类结果
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', marker='o')
    plt.title('DBSCAN Clustering')
    plt.xlabel('Feature 1 (Scaled)')
    plt.ylabel('Feature 2 (Scaled)')
    plt.colorbar(label='Cluster Label')
    plt.show()

    # 打印聚类结果
    print("Cluster labels:")
    print(labels)
    print("\nNumber of clusters:", len(set(labels)) - (1 if -1 in labels else 0))
    print("\nNoise points:", np.sum(labels == -1))


if __name__ == '__main__':
    main()
