import matplotlib.pyplot as plt
import numpy as np

# 挑选四个最具代表性的 Rate：无损/稠密、中度、高度、极度稀疏
rates = ['$\epsilon = 0.0$', '$\epsilon = 0.05$', '$\epsilon = 0.10$', '$\epsilon =  0.15$']

# 数据提取
# 你的完整版最终大小
entro_all = np.array([45.24, 28.96, 12.07, 2.73])
# 没有 Zstd 时的你的算法独立压出的大小
no_zstd = np.array([64.57, 54.40, 17.04, 4.07])
# 原始大小恒为 100
raw = np.array([100.0, 100.0, 100.0, 100.0])

# 计算各部分“消除掉的体积”
# 1. 你的前端算法消除的体积 (100 - no_zstd)
saved_by_frontend = raw - no_zstd
# 2. Zstd 后端消除的额外体积 (no_zstd - entro_all)
saved_by_backend = no_zstd - entro_all
# 3. 最终残留的体积
remaining = entro_all

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 4))

width = 0.5
x = np.arange(len(rates))

# 画堆叠柱状图 (从下往上堆叠)
p1 = ax.bar(x, remaining, width, label='Final Compressed Size', color="#97cc97ff", hatch='////',edgecolor='black')
p2 = ax.bar(x, saved_by_backend, width, bottom=remaining, label='Eliminated by Entropy Coding', color="#b3b3b3ff", alpha=0.8, edgecolor='black')
p3 = ax.bar(x, saved_by_frontend, width, bottom=remaining+saved_by_backend, label='Eliminated by Delta Packing', color="#9eccefff", edgecolor='black')

# 添加数值标签 (展示你的前端有多猛)
for i in range(len(x)):
    # 标注你的前端消除了多少
    ax.text(x[i], remaining[i] + saved_by_backend[i] + saved_by_frontend[i]/2, 
            f"-{saved_by_frontend[i]:.1f}%", ha='center', va='center', color='black', fontweight='bold', fontsize=15)
    
    # 标注 Zstd 消除了多少 (在高度稀疏时 Zstd 作用极小)
    y_center_backend = remaining[i] + saved_by_backend[i]/2
    
    # 【修改点】对于极小值，使用引出线向右上方引出
    if saved_by_backend[i] < 6.0: 
        ax.annotate(f"-{saved_by_backend[i]:.1f}%",
                    xy=(x[i], y_center_backend),                 # 箭头起点：灰色小柱子的中心
                    xytext=(x[i] + 0.05, y_center_backend + 8), # 文字位置：向右平移 0.35，向上拔高 12 (位于蓝色柱体安全区)
                    ha='center', va='center', color='black', fontsize=15,
                    arrowprops=dict(arrowstyle="-", color='black', lw=1))
    else:
        ax.text(x[i], y_center_backend, 
                f"-{saved_by_backend[i]:.1f}%", ha='center', va='center', color='black', fontsize=15)

ax.set_ylabel('Percentage of Original\n Model Size (%)', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(rates, fontsize=15, fontweight='bold')
ax.set_ylim(0, 105)

# 因为改成了向上引出，不再需要特别宽的右侧留白，微调 xlim 防止贴边即可
ax.set_xlim(-0.5, 3.5) 

# 把 Y 轴的刻度字体也设为 15
ax.tick_params(axis='y', labelsize=15)

# 整体给图加一个黑色的边框
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)
    spine.set_visible(True)

# 图例倒序，让 "你的算法" 排在最前面
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='upper right', frameon=True, edgecolor='black', fontsize=15)

plt.tight_layout()
plt.savefig('gain_breakdown.pdf', dpi=300)
print("图1已保存为 gain_breakdown.pdf")