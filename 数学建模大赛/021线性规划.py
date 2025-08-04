import pulp
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. 定义线性规划问题
    prob = pulp.LpProblem("Maximize_Profit", pulp.LpMaximize)

    # 2. 定义变量
    x = pulp.LpVariable('x', lowBound=0)  # x >= 0
    y = pulp.LpVariable('y', lowBound=0)  # y >= 0

    # 3. 定义目标函数
    prob += 4 * x + 3 * y, "Z"

    # 4. 添加约束条件
    prob += 2 * x + y <= 20, "Constraint_1"
    prob += x + 2 * y <= 20, "Constraint_2"

    # 5. 求解问题
    prob.solve()

    # 6. 打印结果
    print("Status:", pulp.LpStatus[prob.status])
    print("Optimal Solution:")
    print(f"x = {x.varValue:.2f}, y = {y.varValue:.2f}")
    print(f"Max Z = {pulp.value(prob.objective):.2f}")

    # 7. 可视化可行域和最优解
    # 定义约束条件的直线
    x_vals = np.linspace(0, 15, 100)
    y1 = 20 - 2 * x_vals  # 2x + y <= 20
    y2 = (20 - x_vals) / 2  # x + 2y <= 20

    # 绘制约束条件
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y1, label=r'$2x + y \leq 20$', color='blue')
    plt.plot(x_vals, y2, label=r'$x + 2y \leq 20$', color='green')

    # 绘制可行域填充
    y_feasible = np.minimum(y1, y2)
    plt.fill_between(x_vals, 0, y_feasible, where=(y_feasible >= 0), color='gray', alpha=0.3, label='Feasible Region')

    # 标记最优解点
    optimal_x = x.varValue
    optimal_y = y.varValue
    plt.scatter(optimal_x, optimal_y, color='red', s=100, label=f'Optimal Point ({optimal_x:.1f}, {optimal_y:.1f})')

    # 添加标签和标题
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Linear Programming: Feasible Region and Optimal Solution', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 15)
    plt.ylim(0, 15)
    plt.show()


if __name__ == '__main__':
    main()
