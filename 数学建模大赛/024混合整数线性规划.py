import pulp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    # 1. 定义并求解混合整数规划
    model = pulp.LpProblem("Mixed_Integer_Programming", pulp.LpMinimize)  # 最小化问题

    # 2. 定义变量
    x = pulp.LpVariable('x', lowBound=0, cat='Continuous')  # 连续变量 x ≥ 0
    y = pulp.LpVariable('y', lowBound=0, upBound=3, cat='Integer')  # 整数变量 y ∈ {0,1,2,3}
    z = pulp.LpVariable('z', cat='Binary')  # 二进制变量 z ∈ {0,1}

    # 3. 定义目标函数和约束
    model += 3 * x + 5 * y + 2 * z, "Total_Cost"
    model += x + y + z >= 10, "Min_Total"
    model += 2 * x + 3 * y + z <= 25, "Max_Resource"

    # 4. 求解问题
    model.solve()

    # 输出结果
    print("Status:", pulp.LpStatus[model.status])
    print("Optimal Solution:")
    print(f"x = {x.varValue:.2f} (Continuous)")
    print(f"y = {int(y.varValue)} (Integer)")
    print(f"z = {int(z.varValue)} (Binary)")
    print(f"Min Z = {pulp.value(model.objective):.2f}")

    # 5. 可视化（3D）
    # 生成所有可能的 (y, z) 组合（整数和二进制）
    y_values = range(0, 4)  # y ∈ {0,1,2,3}
    z_values = [0, 1]  # z ∈ {0,1}
    combinations = [(y_val, z_val) for y_val in y_values for z_val in z_values]

    # 计算每个组合对应的 x 值（通过约束推导）
    feasible_points = []
    for y_val, z_val in combinations:
        # 从约束 x ≥ 10 - y - z 和 x ≤ (25 - 3y - z)/2 推导 x 的范围
        x_min = max(0, 10 - y_val - z_val)
        x_max = (25 - 3 * y_val - z_val) / 2
        if x_min <= x_max:  # 检查是否存在可行解
            x_opt = x_min  # 最小化问题，取 x 的最小值
            feasible_points.append((x_opt, y_val, z_val))

    # 提取最优解
    optimal_point = (x.varValue, y.varValue, z.varValue)

    # 3D 可视化
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制所有可行解
    for point in feasible_points:
        ax.scatter(point[0], point[1], point[2], color='blue', s=50, alpha=0.6)

    # 标记最优解
    ax.scatter(optimal_point[0], optimal_point[1], optimal_point[2],
               color='red', s=200, marker='*',
               label=f'Optimal: x={optimal_point[0]:.2f}, y={optimal_point[1]}, z={optimal_point[2]}')

    # 坐标轴设置
    ax.set_xlabel('x (Continuous)', fontsize=12)
    ax.set_ylabel('y (Integer)', fontsize=12)
    ax.set_zlabel('z (Binary)', fontsize=12)
    ax.set_title('Mixed Integer Linear Programming: Feasible Solutions', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True)

    # 调整视角
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
