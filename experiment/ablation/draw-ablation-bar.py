import matplotlib.pyplot as plt
import numpy as np

# ===== 1. 数据准备 (选取 4 个过渡阶段) =====
# scenarios = [
#     'Lossless / Dense\n($\epsilon = 0.0$)', 
#     'Moderate Lossy\n($\epsilon = 0.05$)', 
#     'Sparse (Crossover)\n($\epsilon = 0.10$)', 
#     'Highly Sparse\n($\epsilon = 0.15$)'
# ]

scenarios = [
    '$\epsilon = 0.0$', 
    '$\epsilon = 0.05$', 
    '$\epsilon = 0.10$', 
    '$\epsilon = 0.15$'
]

# only_pfor    = [48.75, 32.83, 12.37, 2.69]  
# only_shuffle = [45.24, 28.96, 12.47, 3.39]  
# entro_all    = [45.24, 28.96, 12.07, 2.73]  
only_pfor    = [49.75, 34.83, 14.37, 2.8]  
only_shuffle = [45.6, 29.96, 15.47, 3.39]  
entro_all    = [45.24, 28.96, 12.07, 2.73]  

x = np.arange(1) 
# 柱子宽度保持精细设定
width = 0.7  

# ===== 2. 画布与样式设置 =====
plt.style.use('seaborn-v0_8-whitegrid')
# 加大画布高度以容纳上下两部分的文字
fig, axes = plt.subplots(1, 4, figsize=(16.2, 7))

colors = ["#FFE185C6", "#79a679", "#84abc8"]
# labels = ['w/o Shuffle (PFOR Only)', 'w/o PFOR (Shuffle Only)', 'EntroDelta (Full Adaptive)']
labels = ['w/o Shuffle (PFOR Only)', 'w/o PFOR (Shuffle Only)', 'EntroDelta']


# 设置各子图 Y 轴范围
y_limits = [(44, 51), (27.5, 36), (11.5, 16), (1.5, 4.6)]

# ===== 3. 循环绘制 4 个子图 =====
for i, ax in enumerate(axes):

    # 绘制柱状图
    offset = 0.09
    ax.bar(x - width - offset, only_pfor[i], width, color=colors[0], edgecolor='black', zorder=3)
    ax.bar(x,         only_shuffle[i], width, color=colors[1], edgecolor='black', zorder=3)
    ax.bar(x + width + offset, entro_all[i], width, color=colors[2], edgecolor='black', hatch='//', zorder=3)
    
    ax.set_ylim(y_limits[i])

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
        spine.set_visible(True)
    
    # 【修改点 1】将标题文字移到下方，使用 set_xlabel 并增加 labelpad
    ax.set_xlabel(scenarios[i], fontsize=30, fontweight='bold', labelpad=12)
    
    # 格式化刻度与网格
    ax.tick_params(axis='y', labelsize=24) 
    ax.set_xticks([]) # 隐藏 X 轴刻度线
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    if i == 0:
        ax.set_ylabel('Compression Ratio (%)', fontsize=30, fontweight='bold', labelpad=20)

# 【修改点 2】将图例移到图表上方 (upper center)
fig.legend(labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02), 
           fontsize=28, frameon=True, edgecolor='black')

# plt.subplots_adjust(wspace=5)

# 【修改点 3】调整布局，rect 参数中 top=0.92 为顶部的图例留出呼吸感
plt.tight_layout(rect=[0, 0, 1, 0.9], w_pad=3)

# 保存结果
plt.savefig('adaptive_necessity_4stages.pdf', dpi=300, bbox_inches='tight')
# plt.savefig('adaptive_necessity_4stages.png', dpi=300, bbox_inches='tight')

print("绘制完成！图例已移至上方，阶段标签已移至下方。")