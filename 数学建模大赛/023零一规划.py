import pulp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product

# 1. 定义并求解 0-1 规划问题
# 初始化问题
prob = pulp.LpProblem("0-1_Programming_3D", pulp.LpMaximize)

# 定义二进制变量
x1 = pulp.LpVariable('x1', cat='Binary')  # x1 ∈ {0, 1}
x2 = pulp.LpVariable('x2', cat='Binary')  # x2 ∈ {0, 1}
x3 = pulp.LpVariable('x3', cat='Binary')  # x3 ∈ {0, 1}

# 定义目标函数和约束
prob += 5 * x1 + 7 * x2 + 3 * x3, "Z"  # 最大化目标
prob += 2 * x1 + 4 * x2 + 3 * x3 <= 8, "Resource_Limit"

# 求解问题
prob.solve()

# 提取最优解
optimal_solution = (int(x1.varValue), int(x2.varValue), int(x3.varValue))
optimal_value = pulp.value(prob.objective)

# 2. 生成所有可行解
# 所有可能的 0-1 组合
all_combinations = list(product([0, 1], repeat=3))

# 筛选满足约束的解
feasible_solutions = []
for combo in all_combinations:
    x1_val, x2_val, x3_val = combo
    if 2 * x1_val + 4 * x2_val + 3 * x3_val <= 8:
        feasible_solutions.append(combo)

# 3. 3D 可视化
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制所有可行解（红色）
for point in feasible_solutions:
    ax.scatter(*point, color='red', s=100, depthshade=True)

# 标记最优解（金色五角星）
ax.scatter(*optimal_solution, color='gold', s=300, marker='*',
           label=f'Optimal Solution: {optimal_solution}\nZ = {optimal_value}')

# 坐标轴标签和标题
ax.set_xlabel('x1', fontsize=14, labelpad=15)
ax.set_ylabel('x2', fontsize=14, labelpad=15)
ax.set_zlabel('x3', fontsize=14, labelpad=15)
ax.set_title('0-1 Programming: Feasible Solutions (Red) vs Optimal (Gold)',
             fontsize=16, pad=20)

# 设置坐标轴范围和刻度
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_zticks([0, 1])

# 添加图例和网格
ax.legend(fontsize=12, loc='upper right')
ax.grid(True)

# 调整视角以便清晰观察
ax.view_init(elev=25, azim=45)

plt.tight_layout()
plt.show()