import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1. 定义目标函数和约束
def objective(x):
    return (x[0] - 2) ** 2 + (x[1] - 3) ** 2  # 目标函数

def constraint(x):
    return x[0] ** 2 + x[1] ** 2 - 4  # 非线性约束 g(x) ≤ 0

# 2. 定义罚函数
def penalty_function(x, rho):
    penalty = max(0, constraint(x)) ** 2  # 二次罚函数
    return objective(x) + rho * penalty

# 3. 罚函数法求解
def solve_with_penalty(rho=1.0, max_iter=100, tol=1e-6):
    x0 = np.array([0.0, 0.0])  # 初始猜测
    history = []

    for _ in range(max_iter):
        # 最小化带罚函数的无约束问题
        res = minimize(lambda x: penalty_function(x, rho), x0, method='BFGS')
        x_opt = res.x
        history.append(x_opt.copy())

        # 检查约束违反程度
        violation = max(0, constraint(x_opt))
        if violation < tol:
            break

        # 增大罚系数（自适应调整）
        rho *= 2
        x0 = x_opt

    return x_opt, history

def main():
    # 4. 求解并可视化
    solution, history = solve_with_penalty()

    # 打印结果
    print("Optimal Solution (x, y):", solution)
    print("Objective Value:", objective(solution))
    print("Constraint Violation:", max(0, constraint(solution)))

    # 绘制优化路径和约束
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制目标函数的等高线
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = objective([X, Y])
    ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)

    # 绘制约束边界（圆）
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(2 * np.cos(theta), 2 * np.sin(theta), 'r-', label='Constraint: $x^2 + y^2 = 4$')

    # 绘制优化路径
    history = np.array(history)
    ax.plot(history[:, 0], history[:, 1], 'bo-', label='Optimization Path')
    ax.scatter(solution[0], solution[1], color='red', s=100, marker='*',
               label=f'Optimal Point: ({solution[0]:.2f}, {solution[1]:.2f})')

    # 标记初始点
    ax.scatter(0, 0, color='black', s=50, label='Initial Guess')

    # 设置图形属性
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Nonlinear Programming with Penalty Method', fontsize=14)
    ax.legend()
    ax.grid(True)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    plt.show()


if __name__ == '__main__':
    main()