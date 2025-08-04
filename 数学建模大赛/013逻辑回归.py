import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    # 1. 生成模拟客户行为数据集
    np.random.seed(42)
    n_samples = 1000
    browsing_time = np.random.randint(1, 50, size=n_samples)
    num_products_viewed = np.random.randint(1, 10, size=n_samples)
    num_items_in_cart = np.random.randint(0, 5, size=n_samples)

    purchased = (browsing_time > 20) & (num_products_viewed > 5) & (num_items_in_cart > 2)
    purchased = purchased.astype(int)

    # 组合成 DataFrame
    df = pd.DataFrame({
        'browsing_time': browsing_time,
        'num_products_viewed': num_products_viewed,
        'num_items_in_cart': num_items_in_cart,
        'purchased': purchased
    })

    # 2. 准备特征和目标值
    X = df[['browsing_time', 'num_products_viewed', 'num_items_in_cart']]
    y = df['purchased']

    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. 使用逻辑回归模型训练
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)

    # 5. 对测试集进行预测
    y_pred = log_reg.predict(X_test)

    # 6. 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    print('模型准确度：', accuracy)

    # 输出混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print('\n混淆矩阵：\n', cm)
    # 输出分类报告
    cr = classification_report(y_test, y_pred)
    print('\n分类报告：\n', cr)

    # 7. 可视化模型效果
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Result', marker='o')
    plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicted Result', marker='x')
    plt.title('Actual Purchase vs Predicted Purchase')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Whether to Purchase (0=No Purchase, 1=Purchase)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
