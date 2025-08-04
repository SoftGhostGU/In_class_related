import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100  # 避免除以0
    return mse, r2, rmse, mae, mape

def main():
    # 1. 生成模拟的股票数据
    np.random.seed(42)
    num_days = 1000  # 模拟数据的天数

    # 生成时间序列
    dates = pd.date_range(start="2023-01-01", periods=num_days, freq="D")

    # 生成股票价格
    def generate_stock_prices(num_days):
        # 初始化价格，第一个价格为随机数
        prices = [np.random.uniform(100, 200)]  # 初始价格范围在100到200之间
        for _ in range(1, num_days):
            # 每天的价格基于前一天的价格随机波动
            change = np.random.normal(0, 2)  # 每天的变化范围，均值为0，标准差为2
            prices.append(prices[-1] + change)
        return prices

    # 生成模拟的股票价格
    stock_prices = generate_stock_prices(num_days)

    # 生成成交量（模拟为随时间波动的正态分布）
    volume = np.random.normal(10000, 5000, num_days)

    # 创建 DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Close': stock_prices,
        'Volume': volume
    })

    # 设置 Date 为索引
    df.set_index('Date', inplace=True)

    # 2. 特征工程
    # 提取特征（例如收盘价的移动平均线、成交量等）
    df['Volume_MA'] = df['Volume'].rolling(window=30).mean()  # 成交量的30天移动平均
    df['Close_MA'] = df['Close'].rolling(window=30).mean()  # 收盘价的30天移动平均
    df['Close_pct_change'] = df['Close'].pct_change()  # 日收益率
    df = df.dropna()  # 去掉缺失值（滚动窗口会引入NA值）

    # 定义特征和目标变量
    X = df[['Volume', 'Volume_MA', 'Close_MA', 'Close_pct_change']]
    y = df['Close']  # 预测目标：收盘价

    # 3. 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # 避免打乱时间顺序

    # 4. 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. 定义参数网格
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # 6. 使用 GridSearchCV 进行参数优化
    rf_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=5,  # 5 折交叉验证
        scoring='neg_mean_squared_error',  # 使用均方误差的负值作为评分标准
        verbose=1,  # 显示搜索过程
        n_jobs=-1  # 使用所有 CPU 核心
    )
    grid_search.fit(X_train_scaled, y_train)

    # 获取最佳模型及其参数
    best_rf = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # 输出最佳参数
    print("Best Parameters:", best_params)

    # 7. 模型预测
    y_train_pred = best_rf.predict(X_train_scaled)
    y_test_pred = best_rf.predict(X_test_scaled)

    # 8. 模型评估
    # 计算训练集和测试集的评估指标
    train_mse, train_r2, train_rmse, train_mae, train_mape = evaluate_model(y_train, y_train_pred)
    test_mse, test_r2, test_rmse, test_mae, test_mape = evaluate_model(y_test, y_test_pred)

    # 输出模型评估结果
    print("\nTraining Set Performance:")
    print(f"MSE: {train_mse:.4f}")
    print(f"R^2: {train_r2:.4f}")
    print(f"RMSE: {train_rmse:.4f}")
    print(f"MAE: {train_mae:.4f}")
    print(f"MAPE: {train_mape:.4f}%")

    print("\nTest Set Performance:")
    print(f"MSE: {test_mse:.4f}")
    print(f"R^2: {test_r2:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"MAPE: {test_mape:.4f}%")

    # 9. 可视化预测结果
    # 可视化训练集预测结果
    plt.figure(figsize=(14, 6))
    plt.plot(df.index[:len(y_train)], y_train, color='blue', label='Actual Price (Train)', linewidth=1.2)
    plt.plot(df.index[:len(y_train)], y_train_pred, color='red', label='Predicted Price (Train)', linewidth=1.2)
    plt.title('Stock Price Prediction using Random Forest Regression - Training Set')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 可视化测试集预测结果
    plt.figure(figsize=(14, 6))
    plt.plot(df.index[-len(y_test):], y_test, color='blue', label='Actual Price (Test)', linewidth=1.2)
    plt.plot(df.index[-len(y_test):], y_test_pred, color='red', label='Predicted Price (Test)', linewidth=1.2)
    plt.title('Stock Price Prediction using Random Forest Regression - Test Set')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 10. 特征重要性分析
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)


if __name__ == '__main__':
    main()