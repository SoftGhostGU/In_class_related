import numpy as np
import matplotlib.pyplot as plt


# 1. 定义目标函数和约束
def objective(x):
    return (x[0] - 2) ** 2 + (x[1] - 3) ** 2


def constraint(x):
    return 3 - x[0] - x[1]  # 约束 x1 + x2 <= 3


# 2. 黏菌算法参数设置
n_slimes = 50  # 黏菌个体数
max_iter = 100  # 最大迭代次数
dim = 2  # 变量维度 (x1, x2)
lb = np.array([0, 0])  # 变量下界
ub = np.array([3, 3])  # 变量上界


# 3. 黏菌算法核心
def slime_mould_algorithm():
    # 初始化黏菌位置
    def initialize_slimes():
        init_slimes = []
        while len(init_slimes) < n_slimes:
            x = np.random.uniform(lb, ub)
            if constraint(x) >= 0:
                init_slimes.append(x)
        return np.array(init_slimes)

    slimes = initialize_slimes()
    fitness = np.array([objective(p) for p in slimes])
    best_fitness = np.min(fitness)
    best_slime = slimes[np.argmin(fitness)]

    history = [best_fitness]

    for iter in range(max_iter):
        if iter % 10 == 0:  # 每10次迭代打印一次
            print(f"Iter {iter}: Best fitness = {best_fitness}")
        # 1. 安全计算arctanh（避免除以零）
        progress = iter / max_iter
        a = np.arctanh(np.clip(1 - progress, -0.99, 0.99))  # 安全计算
        b = 1 - progress

        # 2. 计算适应度权重
        sorted_idx = np.argsort(fitness)
        worst_fitness = np.max(fitness)
        weights = 1 - fitness / (worst_fitness + 1e-8)

        # 3. 更新黏菌位置
        new_slimes = np.zeros_like(slimes)
        for i in range(n_slimes):
            if np.random.rand() < 0.5:  # 探索阶段
                r1, r2 = np.random.rand(), np.random.rand()
                new_slimes[i] = slimes[i] + (ub - lb) * (a * r1 - b * r2)
            else:  # 开发阶段
                p = np.tanh(np.abs(fitness[i] - best_fitness))
                vb = np.random.uniform(-p, p, size=dim)
                if i == 0:
                    new_slimes[i] = best_slime + vb * (weights[i] * slimes[sorted_idx[0]] - slimes[sorted_idx[-1]])
                else:
                    rand_upper = np.random.choice(sorted_idx[:n_slimes // 2])
                    rand_lower = np.random.choice(sorted_idx[n_slimes // 2:])
                    new_slimes[i] = slimes[i] + vb * (slimes[rand_upper] - slimes[rand_lower])

        # 4. 边界处理和约束修正
        new_slimes = np.clip(new_slimes, lb, ub)
        for i in range(n_slimes):
            max_corrections = 10
            while constraint(new_slimes[i]) < 0:
                new_slimes[i] = (new_slimes[i] + best_slime) / 2

        # 5. 更新适应度
        new_fitness = np.array([objective(p) for p in new_slimes])
        improved_idx = new_fitness < fitness
        slimes[improved_idx] = new_slimes[improved_idx]
        fitness[improved_idx] = new_fitness[improved_idx]

        # 6. 更新历史最优
        current_best = np.min(new_fitness)
        if current_best < best_fitness:
            best_fitness = current_best
            best_slime = new_slimes[np.argmin(new_fitness)]
        history.append(best_fitness)

    return best_slime, history


# 4. 运行算法并可视化
best_solution, fitness_history = slime_mould_algorithm()

print(f"Optimal Solution: ({best_solution[0]:.4f}, {best_solution[1]:.4f})")
print(f"Objective Value: {objective(best_solution):.4f}")
print(f"Constraint Violation: {max(0, constraint(best_solution)):.4f}")

# 可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fitness_history)
plt.title("Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")

plt.subplot(1, 2, 2)
x = np.linspace(0, 3, 100)
y = np.linspace(0, 3, 100)
X, Y = np.meshgrid(x, y)
Z = objective([X, Y])
plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
plt.colorbar()
plt.plot(x, 3 - x, 'r-', label='Constraint')
plt.scatter(best_solution[0], best_solution[1], c='red', s=100, marker='*', label='Optimal')
plt.scatter(2, 3, c='blue', s=50, label='Target')
plt.legend()
plt.title("Solution Space")
plt.tight_layout()
plt.show()