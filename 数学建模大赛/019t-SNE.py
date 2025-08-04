# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

def main():
    # 1. 加载鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 2. 使用TSNE进行降维，将4维数据降至2维
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(X)

    # 3. 创建DataFrame，并将降维后的数据添加到其中
    df_tsne = pd.DataFrame(X_tsne, columns=["Dim1", "Dim2"])
    df_tsne["target"] = y

    # 4. 绘制散点图，将不同种类的鸢尾花用不同颜色表示
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b']
    for target, color in zip([0, 1, 2], colors):
        indices_to_keep = df_tsne["target"] == target
        plt.scatter(df_tsne.loc[indices_to_keep, "Dim1"],
                    df_tsne.loc[indices_to_keep, "Dim2"],
                    c=color, label=target)

    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("iris dataset after t-SNE")
    plt.legend(iris.target_names)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
