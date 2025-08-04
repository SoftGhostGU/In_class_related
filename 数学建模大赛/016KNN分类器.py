import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def main():
    # 1. 生成模拟数据集
    X, y = make_classification(
        n_samples=1000,  # 样本数量
        n_features=2,  # 特征数量
        n_informative=2,  # 有用特征数量
        n_redundant=0,  # 冗余特征数量
        n_classes=3,  # 类别数量
        n_clusters_per_class=1,
        random_state=42
    )

    # 2. 将数据集划分为训练集和测试集(80%训练，20%测试)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # 3. 数据标准化处理(KNN对特征尺度敏感)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. 创建KNN分类器，选择K=5(邻居数为5)
    knn = KNeighborsClassifier(n_neighbors=5)

    # 5. 训练KNN模型
    knn.fit(X_train_scaled, y_train)

    # 6. 对测试集进行预测
    y_pred = knn.predict(X_test_scaled)

    # 7. 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.4f}")

    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    conf_mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # 8. 可视化决策边界(仅适用于2D特征)
    if X.shape[1] == 2:
        plt.figure(figsize=(10, 8))

        # 创建网格点
        h = 0.02  # 步长
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # 预测网格点的类别
        Z = knn.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)

        # 绘制决策边界
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Paired)

        # 绘制训练点
        plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train,
                    edgecolors='k', cmap=plt.cm.Paired, s=50, label='Train')

        # 绘制测试点
        plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test,
                    marker='x', cmap=plt.cm.Paired, s=100, label='Test')

        plt.title(f"KNN Classification (k=5)\nAccuracy: {accuracy:.4f}")
        plt.xlabel('Feature 1 (scaled)')
        plt.ylabel('Feature 2 (scaled)')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
