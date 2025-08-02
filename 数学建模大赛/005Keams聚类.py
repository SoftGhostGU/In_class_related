import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def main():
    # 1. 生成随机客户数据（假设每个客户有两个特征）
    np.random.seed(42)  # 保证每次运行生成的数据相同
    num_customers = 200
    num_features = 2  # 假设有 2 个特征（如：购买频率和消费金额）
    data = np.random.rand(num_customers, num_features) * 100  # 随机生成数据，范围在 0 到 100 之间
    print("客户数据形状：", data.shape)

    # 2. 数据标准化（K-Means 对数据的尺度敏感）
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    print("\n标准化后的数据形状：", scaled_data.shape)

    # 3. 使用 KMeans 进行聚类
    # 假设聚成 k=3 类
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)

    # 4. 打印聚类结果
    print("\n聚类标签：")
    print(cluster_labels)
    print("\n聚类中心：")
    print(kmeans.cluster_centers_)

    # 5. 评价聚类效果（使用轮廓系数）
    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
    print(f"\n轮廓系数（Silhouette Score）：{silhouette_avg:.3f}")

    # 6. 可视化聚类结果
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for i in range(max(cluster_labels) + 1):
        plt.scatter(
            scaled_data[cluster_labels == i, 0],
            scaled_data[cluster_labels == i, 1],
            color=colors[i],
            label=f"Cluster {i+1}"
        )
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=300,
        c='yellow',
        marker='X',
        label='Cluster Centers'
    )
    plt.title("K-Means Clustering Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
