import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100  # 避免除以0
    return mse, r2, rmse, mae, mape

def main():
    # 1. 生成模拟房价数据
    np.random.seed(42)
    n_samples = 100

    # 生成特征数据
    data = {
        'Square_Footage': np.random.randint(800, 4000, size=n_samples),
        'Number_of_Rooms': np.random.randint(1, 6, size=n_samples),
        'Age_of_House': np.random.randint(0, 50, size=n_samples),
        'Has_Garden': np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),
        'Price': 0  # 先初始化为0
    }

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 计算房价（基于一些假设的公式）
    df['Price'] = (
            df['Square_Footage'] * 100 +
            df['Number_of_Rooms'] * 50000 +
            (50 - df['Age_of_House']) * 2000 +
            df['Has_Garden'] * 30000 +
            np.random.normal(0, 20000, size=n_samples)
    )

    # 2. 数据预处理
    # 分割特征和目标变量
    X = df.drop('Price', axis=1)
    y = df['Price']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. 创建决策树回归模型
    dt_model = DecisionTreeRegressor(random_state=42)

    # 定义参数网格
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # 使用GridSearchCV进行参数优化
    grid_search = GridSearchCV(estimator=dt_model,
                               param_grid=param_grid,
                               cv=5,
                               scoring='neg_mean_squared_error',
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # 获取最佳模型及其参数
    best_dt = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # 4. 模型评估
    # 在训练集和测试集上进行预测
    y_train_pred = best_dt.predict(X_train)
    y_test_pred = best_dt.predict(X_test)

    # 计算评估指标
    train_mse, train_r2, train_rmse, train_mae, train_mape = evaluate_model(y_train, y_train_pred)
    test_mse, test_r2, test_rmse, test_mae, test_mape = evaluate_model(y_test, y_test_pred)

    # 5. 输出结果 - 按照您图片中的格式
    print("Best model performance on training set:")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Train MAPE: {train_mape:.4f}%")

    print("\nBest model performance on test set:")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test MAPE: {test_mape:.4f}%")

    print("\nBest model configuration:", best_params)

    # 6. 特征重要性分析
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_dt.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

    # 7. 可视化实际值与预测值
    # 训练集可视化
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(y_train)), y_train, color='blue', label='Actual Values', linewidth=1.5)
    plt.plot(np.arange(len(y_train)), y_train_pred, color='red', label='Predicted Values', linewidth=1.5)
    plt.title('Decision Tree Regression Prediction on Training Set')
    plt.xlabel('Index')
    plt.ylabel('Target Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 测试集可视化
    plt.plot(np.arange(len(y_test)), y_test, color='blue', label='Actual Values', linewidth=1.5)
    plt.plot(np.arange(len(y_test)), y_test_pred, color='red', label='Predicted Values', linewidth=1.5)
    plt.title('Decision Tree Regression Prediction on Test Set')
    plt.xlabel('Index')
    plt.ylabel('Target Value')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()