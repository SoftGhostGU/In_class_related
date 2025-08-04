import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

# 1. 生成非线性可分数据集
X, y = make_circles(n_samples=500, noise=0.1, factor=0.2, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 2. 定义三种核函数的参数搜索空间
param_grids = {
    "RBF Kernel": {
        'kernel': ['rbf'],
        'C': [0.1, 1, 10, 100],
        'gamma': [0.01, 0.1, 1, 10]
    },
    "Polynomial Kernel": {
        'kernel': ['poly'],
        'C': [0.1, 1, 10],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto'],
        'coef0': [0, 1]
    },
    "Sigmoid Kernel": {
        'kernel': ['sigmoid'],
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1],
        'coef0': [0, 1]
    }
}

# 3. 进行参数调优并训练最佳模型
best_models = {}
results = {}

print("开始参数调优...")
for name, param_grid in param_grids.items():
    print(f"\n正在调优 {name}...")
    grid = GridSearchCV(
        svm.SVC(),
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    # 保存最佳模型
    best_models[name] = grid.best_estimator_
    results[name] = {
        'best_params': grid.best_params_,
        'best_score': grid.best_score_,
        'test_accuracy': accuracy_score(y_test, grid.predict(X_test))
    }

    print(f"最佳参数: {grid.best_params_}")
    print(f"交叉验证最佳得分: {grid.best_score_:.4f}")
    print(f"测试集准确率: {results[name]['test_accuracy']:.4f}")

# 4. 可视化比较
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for (name, model), ax in zip(best_models.items(), axes):
    # 可视化决策边界
    DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        cmap=plt.cm.Paired,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        shading="auto",
        alpha=0.3,
    )

    # 绘制数据点和支持向量
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    ax.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=100,
        facecolors="none",
        edgecolors="k",
        linewidths=1.5,
    )
    ax.set_title(
        f"{name}\n"
        f"Params: {str(results[name]['best_params'])}\n"
        f"Test Acc: {results[name]['test_accuracy']:.4f}"
    )

plt.tight_layout()
plt.show()

# 5. 打印最终结果比较
print("\n最终模型比较:")
for name, res in results.items():
    print(f"\n{name}:")
    print(f"最佳参数: {res['best_params']}")
    print(f"交叉验证最佳得分: {res['best_score']:.4f}")
    print(f"测试集准确率: {res['test_accuracy']:.4f}")