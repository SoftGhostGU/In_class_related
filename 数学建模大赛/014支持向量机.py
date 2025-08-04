# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from hanlp.metrics.chunking.sequence_labeling import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


def main():
    # 1. 加载鸢尾花数据集
    iris = load_iris()
    X = iris.data  # 特征矩阵
    y = iris.target  # 目标变量

    # 2. 查看数据集的基本信息
    print("数据集特征数量:", X.shape[1])
    print("数据集样本数量:", X.shape[0])
    print("类别数量:", len(np.unique(y)))
    print("类别名称:", iris.target_names)

    # 3. 数据分割：训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4.1. 使用支持向量机 (SVM) 进行分类 - 线性核函数
    model_linear = SVC(kernel='linear', random_state=42)
    model_linear.fit(X_train, y_train)

    # 4.2. 使用支持向量机 (SVM) 进行分类 - rbf核函数
    model_rbf = SVC(kernel='rbf', C=1, gamma='auto', random_state=42)
    model_rbf.fit(X_train, y_train)

    # 5.1. 在测试集上进行预测 - 线性核函数
    y_pred_linear = model_linear.predict(X_test)

    # 5.2. 在测试集上进行预测 - rbf核函数
    y_pred_rbf = model_rbf.predict(X_test)

    # 6.1. 评估模型性能 - 线性核函数
    print("\n分类报告（线性核函数）：")
    print(classification_report(y_test, y_pred_linear, target_names=iris.target_names))
    print("\n混淆矩阵（线性核函数）：")
    conf_matrix_linear = confusion_matrix(y_test, y_pred_linear)
    print(conf_matrix_linear)
    print("\n准确率（线性核函数）：", accuracy_score(y_test, y_pred_linear))

    # 6.2. 评估模型性能 - rbf核函数
    print("\n\n分类报告（rbf核函数）：")
    print(classification_report(y_test, y_pred_rbf, target_names=iris.target_names))
    print("\n混淆矩阵（rbf核函数）：")
    conf_matrix_rbf = accuracy_score(y_test, y_pred_rbf)
    print(conf_matrix_rbf)

    # 选择两个特征进行可视化（这里选择前两个特征）
    X_vis = X_train[:, :2]  # 只取前两个特征（萼片长度和萼片宽度）

    # 重新训练模型（因为我们只使用两个特征来可视化）
    model_linear_vis = SVC(kernel='linear', random_state=42)
    model_linear_vis.fit(X_vis, y_train)

    model_rbf_vis = SVC(kernel='rbf', C=1, gamma='auto', random_state=42)
    model_rbf_vis.fit(X_vis, y_train)

    # 7. 结果可视化
    plt.figure(figsize=(15, 6))

    # 定义颜色映射
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # 生成网格数据
    h = 0.02  # 步长
    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 线性核函数可视化
    plt.subplot(1, 2, 1)
    Z = model_linear_vis.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_train, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('SVM with Linear Kernel')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])

    # RBF核函数可视化
    plt.subplot(1, 2, 2)
    Z = model_rbf_vis.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_train, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('SVM with RBF Kernel')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
