import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AffinityPropagation

def main():
    # 1. 生成二维数据集，包含多个簇
    centers = [[1, 1], [-1, -1], [1, -1]]  # 定义簇中心
    X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, random_state=42)

    # 2. 使用传播聚类进行聚类
    ap = AffinityPropagation(random_state=42)  # 初始化 Affinity Propagation 模型
    ap.fit(X)  # 对数据进行聚类

    # 获取聚类结果
    labels = ap.labels_  # 每个数据点的簇标签
    cluster_centers_indices = ap.cluster_centers_indices_  # 代表点的索引
    n_clusters = len(cluster_centers_indices)  # 簇的数量

    # 3. 输出结果
    print("Cluster labels: \n", labels)
    print("\nCluster centers indices:", cluster_centers_indices)
    print("\nNumber of clusters:", n_clusters)

    # 4. 可视化聚类结果
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(X[cluster_centers_indices, 0], X[cluster_centers_indices, 1], s=200, c='red', marker='*',
                label='Cluster Centers')
    plt.title('Affinity Propagation Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
