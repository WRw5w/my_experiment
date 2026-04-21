import math


def calculate_nesterov_momentum(steps):
    t_prev = 1.0  # t_0
    results = {}

    # 我们需要计算到第 50 步，所以循环 50 次
    for k in range(1, max(steps) + 1):
        # 核心递推公式：t_k^2 - t_k = t_{k-1}^2
        # 解方程取正根：t_k = (1 + sqrt(1 + 4 * t_{k-1}^2)) / 2
        t_curr = (1 + math.sqrt(1 + 4 * t_prev ** 2)) / 2

        # 动量系数 beta_k = (t_{k-1} - 1) / t_k
        beta_k = (t_prev - 1) / t_curr

        if k in steps:
            results[k] = (t_curr, beta_k)

        t_prev = t_curr

    return results


# 目标步数
target_steps = [1, 2, 5, 10, 50]
data = calculate_nesterov_momentum(target_steps)

print(f"{'步数 (k)':<10} | {'t_k (放大系数)':<15} | {'beta_k (动量系数)':<15}")
print("-" * 45)
for k in target_steps:
    tk, bk = data[k]
    print(f"{k:<10} | {tk:<15.6f} | {bk:<15.6f}")