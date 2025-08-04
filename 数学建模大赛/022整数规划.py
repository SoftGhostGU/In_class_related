import pulp
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. 定义整数规划问题
    prob = pulp.LpProblem("Maximize_Profit_Integer", pulp.LpMaximize)

    # 2. 定义整数变量
    x = pulp.LpVariable('x', lowBound=0, cat='Integer')  # x >= 0 且为整数
    y = pulp.LpVariable('y', lowBound=0, cat='Integer')  # y >= 0 且为整数

    # 3. 定义目标函数和约束
    prob += 4 * x + 3 * y, "Z"
    prob += 2 * x + y <= 20, "Constraint_1"
    prob += x + 2 * y <= 20, "Constraint_2"

    # 4. 求解问题
    prob.solve()

    # 5. 打印结果
    print("Status:", pulp.LpStatus[prob.status])
    print("Optimal Solution (Integer):")
    print(f"x = {int(x.varValue)}, y = {int(y.varValue)}")
    print(f"Max Z = {pulp.value(prob.objective)}")

    # 6. 可视化整数可行解
    plt.figure(figsize=(10, 6))

    # 绘制约束条件
    x_vals = np.linspace(0, 15, 100)
    y1 = 20 - 2 * x_vals  # 2x + y <= 20
    y2 = (20 - x_vals) / 2  # x + 2y <= 20
    plt.plot(x_vals, y1, label=r'$2x + y \leq 20$', color='blue')
    plt.plot(x_vals, y2, label=r'$x + 2y \leq 20$', color='green')

    # 绘制整数可行点
    feasible_points = []
    for x_int in range(0, 11):
        for y_int in range(0, 11):
            if 2 * x_int + y_int <= 20 and x_int + 2 * y_int <= 20:
                feasible_points.append((x_int, y_int))
                plt.scatter(x_int, y_int, color='black', s=30, alpha=0.5)

    # 标记最优整数解
    optimal_x = int(x.varValue)
    optimal_y = int(y.varValue)
    plt.scatter(optimal_x, optimal_y, color='red', s=100, label=f'Optimal Integer Point ({optimal_x}, {optimal_y})')

    # 标签和标题
    plt.xlabel('x (Integer)', fontsize=12)
    plt.ylabel('y (Integer)', fontsize=12)
    plt.title('Integer Programming: Feasible Integer Solutions', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.show()


if __name__ == '__main__':
    main()