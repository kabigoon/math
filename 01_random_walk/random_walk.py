#!/usr/bin/env python3
"""随机游走：为什么误差按 √N 增长，而不是 N。

一次"醉汉走路"：从 0 出发，每步随机 ±1。走 N 步之后离家多远？

    理论：E[X_N] = 0,  Var[X_N] = N,  std = √N,  E|X_N| = √(2N/π) ≈ 0.798·√N
    对照：如果每步都朝同一方向（确定性），距离就是 N

脚本做四件事：
  1. 一条示例路径（ASCII 轨迹）
  2. 终点分布（ASCII 直方图）
  3. N 从 16 到 4096：实测 mean|x| 与 std 对照公式，看比值是否收敛到 1
  4. 与确定性行走对比：同样 N 步，一个 ≈0.8√N，一个是 N

用法：
    python3 random_walk.py
    python3 random_walk.py --trials 50000 --max-steps 8192 --seed 7
    python3 random_walk.py --csv results/random_walk.csv
"""

import argparse
import csv
import math
import pathlib

import numpy as np


def simulate(trials: int, steps: int, rng: np.random.Generator) -> np.ndarray:
    """返回 (trials, steps) 的位置矩阵：每一行是一条路径每步之后的位置。"""
    moves = rng.choice([-1, 1], size=(trials, steps)).astype(np.int32)
    return np.cumsum(moves, axis=1)


def ascii_path(pos: np.ndarray, rows: int = 7) -> str:
    """把一条路径画成 ASCII，横轴是步数。"""
    lo, hi = int(pos.min()), int(pos.max())
    span = max(1, hi - lo)
    grid = [[" "] * len(pos) for _ in range(rows)]
    for x, p in enumerate(pos):
        r = rows - 1 - int((p - lo) / span * (rows - 1))
        grid[r][x] = "*"
    return "\n".join("  " + "".join(row) for row in grid)


def ascii_hist(values: np.ndarray, bins: int = 15, width: int = 36) -> str:
    counts, edges = np.histogram(values, bins=bins)
    peak = max(1, counts.max())
    lines = []
    for i in range(bins):
        bar = "█" * int(round(width * counts[i] / peak))
        lines.append(f"  {edges[i]:>8.1f} | {bar} {counts[i]}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=20000, help="模拟多少条路径")
    ap.add_argument("--max-steps", type=int, default=4096, help="最长走多少步")
    ap.add_argument("--seed", type=int, default=0)
    default_csv = pathlib.Path(__file__).resolve().parent / "results" / "random_walk.csv"
    ap.add_argument("--csv", default=str(default_csv), help="把表格写成 CSV（空字符串则跳过）")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pos = simulate(args.trials, args.max_steps, rng)

    # ---- 1. 一条示例路径 ----
    print("=== 1. 一条示例路径（前 80 步）===")
    print(ascii_path(pos[0, :80]))
    print(f"  80 步后位置 = {pos[0, 79]:+d}    （离原点只有个位数）")

    # ---- 2. 终点分布 ----
    final = pos[:, -1]
    print(f"\n=== 2. {args.trials} 条路径走 {args.max_steps} 步后的终点分布 ===")
    print(ascii_hist(final))
    print(f"  实测: mean|X| = {np.abs(final).mean():.2f}   std = {final.std():.2f}")
    print(f"  理论: E|X| = √(2N/π) = {math.sqrt(2 * args.max_steps / math.pi):.2f}   "
          f"std = √N = {math.sqrt(args.max_steps):.2f}")

    # ---- 3. √N 缩放 ----
    print("\n=== 3. 缩放：实测 / 理论 是否收敛到 1 ===")
    print("     N |   mean|X| |  √(2N/π) | 比值 |     std |    √N | 比值")
    rows = []
    for n in (16, 64, 256, 1024, 4096):
        if n > args.max_steps:
            continue
        x = pos[:, n - 1]
        mean_abs, std = float(np.abs(x).mean()), float(x.std())
        th_mean, th_std = math.sqrt(2 * n / math.pi), math.sqrt(n)
        print(f"  {n:>5} | {mean_abs:>9.2f} | {th_mean:>8.2f} | {mean_abs / th_mean:>4.2f} "
              f"| {std:>7.2f} | {th_std:>5.2f} | {std / th_std:>4.2f}")
        rows.append([n, mean_abs, std, th_mean, th_std])

    # ---- 4. 与确定性行走对比 ----
    n = args.max_steps
    print(f"\n=== 4. 同样是 {n} 步：随机 vs 确定性 ===")
    print(f"  随机游走   典型距离 ≈ {np.abs(final).mean():.1f}   (0.798·√N)")
    print(f"  一路向前   距离     = {n}   (N)")
    print(f"  倍数差     ≈ {n / max(1e-9, np.abs(final).mean()):.0f}×")

    # ---- 5. 方差相加 ----
    print("\n=== 5. 为什么是 √N：方差可加，标准差不可加 ===")
    for n in (16, 64, 256, 1024):
        if n > args.max_steps:
            continue
        x = pos[:, n - 1]
        print(f"  N={n:>4}: 实测 Var = {x.var():>8.2f}   （理论 = N = {n}）；"
              f"std = {x.std():>6.2f} ≠ N")
    print("  → Var 随 N 线性增长，开方后 std 就只按 √N 涨")

    if args.csv:
        path = pathlib.Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["N", "mean_abs", "std", "theory_mean_abs", "theory_std"])
            w.writerows(rows)
        print(f"\n表格已写入 {path}")

    print("\n一句话：距离的方差每步 +1，所以 N 步后方差是 N、标准差是 √N。")
    print("        累加 N 次舍入误差是同一个数学：误差 std ≈ 0.5ULP·√N。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
