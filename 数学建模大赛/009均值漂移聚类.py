import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

def main():
    # 1. 生成模拟数据，包含多个簇
    centers = [[1, 1], [-1, -1], [1, -1], [2, 2]]
    X, _ = make_blobs(n_samples=750, centers=centers, cluster_std=0.6)

    # 2. 自动估计带宽参数
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

    # 3. 使用均值漂移算法进行聚类
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    # 4. 获取簇的数量
    n_clusters_ = len(np.unique(labels))

    # 5. 绘制聚类结果
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, alpha=0.5, label='Cluster Centers')
    plt.title('Mean Shift Clustering')
    plt.xlabel('Feature 1 (Scaled)')
    plt.ylabel('Feature 2 (Scaled)')
    plt.colorbar(label='Cluster Label')
    plt.legend()
    plt.show()

    # 6. 输出聚类结果
    print("Cluster labels:\n", labels)
    print("\nNumber of clusters:", len(np.unique(labels)))
    print("\nCluster centers:\n", cluster_centers)


if __name__ == '__main__':
    main()
