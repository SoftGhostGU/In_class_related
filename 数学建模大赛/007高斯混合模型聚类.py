import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

def plot_ellipse(mean, cov, ax):
    v, w = np.linalg.eigh(cov)  # 特征值分解
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # 放大椭圆的半径
    u = w[0] / np.linalg.norm(w[0])  # 椭圆方向

    angle = np.arctan(u[1] / u[0])  # 椭圆旋转的角度
    angle = 180 + 180.0 * angle / np.pi  # 弧度制转为角度制

    ell = Ellipse(mean, v[0], v[1], color='red', alpha=0.3)
    ax.add_patch(ell)


def main():
    # 1. 生成环形分布的数据
    np.random.seed(42)
    n_samples = 1000  # 数据点数量
    n_components = 3  # 环形的层数量
    cluster_std = [1.0, 2.5, 0.5]  # 每层的噪声程度（标准差）

    # 使用 make_blobs 生成数据
    data, true_labels = make_blobs(
        n_samples=n_samples,
        cluster_std=cluster_std,
        random_state=42
    )

    # 2. 使用 GMM 进行聚类
    # 假设聚成 k=4 类（4层环形分布）
    gmm = GaussianMixture(
        n_components=n_components,  # 聚类的数量
        covariance_type='full',  # 允许每个高斯分布有完全的协方差矩阵
        random_state=42
    )
    gmm.fit(data)
    predicted_labels = gmm.predict(data)

    # 3. 输出 GMM 的参数
    print("聚类中心：")
    print(gmm.means_)
    print("\n协方差矩阵：")
    print(gmm.covariances_)
    print("\n每个高斯分布的权重：")
    print(gmm.weights_)

    # 4. 打印聚类结果
    print("聚类标签：", predicted_labels)

    # 5. 可视化结果
    plt.figure(figsize=(12, 5))

    # 可视化原始数据和真实标签
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], c=true_labels, cmap='viridis', s=30, label="True Labels")
    plt.title("True Labels")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    # 可视化 GMM 聚类结果
    plt.subplot(1, 2, 2)
    plt.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis', s=30, label="GMM Clustering")
    ax = plt.gca()
    for mean, cov in zip(gmm.means_, gmm.covariances_):
        plot_ellipse(mean, cov, ax) # 画出每个高斯分布的椭圆
    plt.title("GMM Clustering Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
