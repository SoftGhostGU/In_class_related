import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_moons

def main():
    # 1. 生成两个月牙形状的数据
    np.random.seed(42)  # 保证每次运行生成的数据相同
    num_samples = 300  # 样本数量
    noise = 0.05  # 噪声程度
    data, true_labels = make_moons(n_samples=num_samples, noise=noise, random_state=42)
    print("数据形状：", data.shape)

    # 2. 数据标准化（K-Means 对数据的尺度敏感）
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    print("\n标准化后的数据形状：", scaled_data.shape)

    # 3. 使用光谱聚类进行聚类
    # 假设聚成 k=2 类（两个月牙形）
    n_clusters = 2
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='rbf',
        n_neighbors=20,  # 增加邻居数量
        random_state=42
    )
    cluster_labels = spectral.fit_predict(scaled_data)

    # 4. 打印聚类结果
    print("\n聚类标签：")
    print(cluster_labels)

    # 5. 评价聚类效果（使用轮廓系数）
    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
    print(f"\n轮廓系数（Silhouette Score）：{silhouette_avg:.3f}")

    # 6. 可视化聚类结果
    plt.figure(figsize=(12, 5))

    # 可视化原始数据和真实标签
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], c=true_labels, cmap='viridis', s=30, label="True Labels")
    plt.title("True Labels")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    # 可视化光谱聚类结果
    plt.subplot(1, 2, 2)
    plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap='viridis', s=30, label="Spectral Clustering")
    plt.title("Spectral Clustering Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
