import json
import matplotlib.pyplot as plt

def main():
    # 1. 读取 JSON 数据 (您的压缩方法结果)
    # json_file_path = 'parsed_metrics.json'
    json_file_path = 'parsed_metrics-llama-3.1-8b.json'
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: 找不到文件 {json_file_path}，请确保它与此脚本在同一目录下。")
        return

    # 从 JSON 提取所需的数据
    ours_sizes = [item['Size (%)'] for item in data]
    ours_mmlu = [item['mmlu/acc'] for item in data]
    lossy_rates = [item['Rate'] for item in data]

    # 2. 硬编码基线模型数据
    # LLama-3.1-8b
    gptq_size = 25.0
    gptq_mmlu = 0.6402

    awq_size = 25.0
    awq_mmlu = 0.6665

    gzip_size = 78.94
    gzip_mmlu = 0.6812

    zlib_size = 78.94
    zlib_mmlu = 0.6812

    fm_delta_size = 60.74
    fm_delta_mmlu = 0.6812

    zstd_size = 77.54
    zstd_mmlu = 0.6812

    

    # 3. 全局样式设置 (已全面放大字号以适应 Paper 缩放)
    plt.rcParams.update({
        'font.size': 18,           
        'axes.labelsize': 22,      
        'axes.titlesize': 22,      
        'xtick.labelsize': 18,     
        'ytick.labelsize': 18,     
        'legend.fontsize': 16,     
        'font.family': 'serif', 
    })

    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

    # 4. 绘制“断层连接线” (Visual Bridge)
    ax.plot([ours_sizes[0], fm_delta_size, zstd_size, gzip_size],
            [ours_mmlu[0], fm_delta_mmlu, zstd_mmlu, gzip_mmlu],
            linestyle='--', color='gray', linewidth=1.5, alpha=0.5, zorder=1)
    
    # ⭐ 核心修复 1：将 Baseline 文字大幅向左移动至 X=55 的空白安全区，彻底避开 zlib/gzip
    ax.annotate('Lossless Compression\n Baseline', xy=(60, 0.6812), xytext=(61, 0.6835),
                fontsize=16, color='dimgray', style='italic', ha='center')

    # 5. 绘制核心数据点与曲线
    ax.plot(ours_sizes, ours_mmlu, marker='o', markersize=8, linestyle='-', linewidth=3.0, 
            color="#62a2d3", label=r'EntroDelta (Variable $\epsilon$)', zorder=3)
            
    # 无损点标记尺寸等比放大
    ax.scatter(ours_sizes[0], ours_mmlu[0], color='#62a2d3', marker='o', s=80, 
               edgecolors='black', linewidths=1.5, label=r'EntroDelta (Lossless, $\epsilon$=0.0)', zorder=5)

    # 绘制其他基线散点图
    ax.scatter(gptq_size, gptq_mmlu, marker='^', s=200, color="#59ac59", label='GPTQ 4-bit', zorder=4)
    ax.scatter(awq_size, awq_mmlu, marker='*', s=250, color="#f8a154", label='AWQ 4-bit', zorder=4)
    ax.scatter(gzip_size, gzip_mmlu, marker='X', s=150, color='#F7CF49', label='zlib/gzip (Lossless)', zorder=4)
    ax.scatter(zstd_size, zstd_mmlu, marker='P', s=150, color='#cc98fd', label='zstd (Lossless)', zorder=4)
    # ax.scatter(fm_delta_size, fm_delta_mmlu, marker='D', s=100, color='#8c564b', label='FMdelta', zorder=4)

    # 6. 添加标注 (⭐ 核心修复 2：彻底移除灰线，采用极近距离贴身绝对像素偏移)
    key_rates = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15]

    for i, rate in enumerate(lossy_rates):
        if rate in key_rates:
            txt = f'$\\epsilon$={rate}'
            
            # 使用 offset points 绝对像素偏移，距离缩短至 20 像素左右，形成齿轮般紧凑的无重叠排版
            if rate == 0.0:
                # 文字较长，向右多偏移一点，向下避开折线
                x_offset, y_offset = 75, -20
                txt = r'Lossless ($\epsilon$=0.0)' 
            elif rate == 0.01:
                # 紧贴正上方
                x_offset, y_offset = 0, 20
            elif rate == 0.02:
                # 紧贴正下方
                x_offset, y_offset = 0, -20
            elif rate == 0.03:
                # 紧贴左上方
                x_offset, y_offset = -17, 20
            elif rate == 0.04:
                # 紧贴左下方
                x_offset, y_offset = -15, -20
            elif rate == 0.05:
                # 紧贴左上方
                x_offset, y_offset = -30, 20
            elif rate == 0.1:
                # 紧贴右下方
                x_offset, y_offset = 25, -20
            elif rate == 0.12:
                # 紧贴左上方
                x_offset, y_offset = -25, 20
            elif rate == 0.15:
                # 紧贴右下方
                x_offset, y_offset = 35, -20
            else:
                x_offset, y_offset = 15, 15
            
            text_color = '#1f77b4'
            
            # ⭐ 彻底移除了 arrowprops 逻辑，只保留纯文本标注，并用 offset points 锁死间距
            ax.annotate(txt, 
                        xy=(ours_sizes[i], ours_mmlu[i]), 
                        xytext=(x_offset, y_offset),
                        textcoords='offset points',
                        fontsize=16, color=text_color, fontweight='bold', 
                        ha='center', va='center')

    # 基线文本提示微调
    ax.annotate('GPTQ', (gptq_size, gptq_mmlu), xytext=(gptq_size + 1.5, gptq_mmlu - 0.0015),
                fontsize=16, color="#59ac59", fontweight='bold')
    ax.annotate('AWQ', (awq_size, awq_mmlu), xytext=(awq_size + 1.5, awq_mmlu),
                fontsize=16, color="#f8a154", fontweight='bold')
    
    ax.annotate('zlib/gzip', (gzip_size, gzip_mmlu), xytext=(gzip_size-1, gzip_mmlu + 0.002),
                fontsize=16, color='#F7CF49', fontweight='bold', ha='center')
    ax.annotate('zstd', (zstd_size, zstd_mmlu), xytext=(zstd_size-0.5, zstd_mmlu - 0.0035),
                fontsize=16, color='#cc98fd', fontweight='bold', ha='center')

    # 7. 图表细节美化
    ax.set_xlabel('Compression Ratio  (%)')
    ax.set_ylabel('MMLU Accuracy')

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0.0, 82.0) 
    
    # 因为去掉了高高拔起的引线，我们可以稍微将天花板调平一些，让曲线更加居中饱满
    ax.set_ylim(0.630, 0.692)

    # 图例设置
    ax.legend(loc='lower right', frameon=True, shadow=False, edgecolor='black')

    # 8. 保存并展示
    output_filename = 'pareto_frontier_llama3_1_8b_mmlu.pdf'
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"图表已成功生成并保存为: {output_filename}")

if __name__ == "__main__":
    main()

# import json
# import matplotlib.pyplot as plt

# def main():
#     # 1. 读取 JSON 数据 (您的压缩方法结果)
#     json_file_path = 'parsed_metrics.json'
#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#     except FileNotFoundError:
#         print(f"Error: 找不到文件 {json_file_path}，请确保它与此脚本在同一目录下。")
#         return

#     # 从 JSON 提取所需的数据
#     ours_sizes = [item['Size (%)'] for item in data]
#     ours_mmlu = [item['mmlu/acc'] for item in data]
#     lossy_rates = [item['Rate'] for item in data]

#     # 2. 硬编码基线模型数据
#     gptq_size = 25.0
#     gptq_mmlu = 0.6402

#     awq_size = 25.0
#     awq_mmlu = 0.6665

#     gzip_size = 78.94
#     gzip_mmlu = 0.6812

#     zlib_size = 78.94
#     zlib_mmlu = 0.6812

#     fm_delta_size = 60.74
#     fm_delta_mmlu = 0.6812

#     zstd_size = 77.54
#     zstd_mmlu = 0.6812

#     # 3. 全局样式设置 (⭐ 已全面放大字号以适应 Paper 缩放)
#     plt.rcParams.update({
#         'font.size': 18,           # 从 12 放大到 18
#         'axes.labelsize': 22,      # 从 14 放大到 22 (X/Y轴标题)
#         'axes.titlesize': 22,      # 从 14 放大到 22
#         'xtick.labelsize': 18,     # 从 12 放大到 18 (坐标轴刻度数字)
#         'ytick.labelsize': 18,     # 从 12 放大到 18
#         'legend.fontsize': 16,     # 从 12 放大到 16 (图例文字)
#         'font.family': 'serif', 
#     })

#     fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

#     # 4. 绘制“断层连接线” (Visual Bridge)
#     ax.plot([ours_sizes[0], fm_delta_size, zstd_size, gzip_size],
#             [ours_mmlu[0], fm_delta_mmlu, zstd_mmlu, gzip_mmlu],
#             linestyle='--', color='gray', linewidth=1.5, alpha=0.5, zorder=1)
    
#     # ⭐ 标注字号放大
#     ax.annotate('Lossless Compression Baseline', xy=(63, 0.6812), xytext=(63, 0.6845),
#                 fontsize=16, color='dimgray', style='italic', ha='center')

#     # 5. 绘制核心数据点与曲线
#     # 先画主体蓝色折线，将 markersize 适当放大以匹配大字号
#     ax.plot(ours_sizes, ours_mmlu, marker='o', markersize=8, linestyle='-', linewidth=3.0, 
#             color="#62a2d3", label=r'EntroDelta (Variable $\epsilon$)', zorder=3)
            
#     # 无损点标记尺寸等比放大
#     ax.scatter(ours_sizes[0], ours_mmlu[0], color='#62a2d3', marker='o', s=80, 
#                edgecolors='black', linewidths=1.5, label=r'EntroDelta (Lossless, $\epsilon$=0.0)', zorder=5)

#     # 绘制其他基线散点图
#     ax.scatter(gptq_size, gptq_mmlu, marker='^', s=200, color="#59ac59", label='GPTQ 4-bit', zorder=4)
#     ax.scatter(awq_size, awq_mmlu, marker='*', s=250, color="#f8a154", label='AWQ 4-bit', zorder=4)
#     ax.scatter(gzip_size, gzip_mmlu, marker='X', s=150, color='#F7CF49', label='zlib/gzip', zorder=4)
#     ax.scatter(zstd_size, zstd_mmlu, marker='P', s=150, color='#cc98fd', label='zstd', zorder=4)
#     # ax.scatter(fm_delta_size, fm_delta_mmlu, marker='D', s=100, color='#8c564b', label='FMdelta', zorder=4)

#     # 6. 添加标注 (引线标注防重叠系统)
#     # key_rates = [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15]
#     key_rates = [0.0,  0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15]


#     for i, rate in enumerate(lossy_rates):
#         if rate in key_rates:
#             txt = f'$\\epsilon$={rate}'
            
#             # 使用引线属性
#             arrow_props = None
            
#             if rate == 0.0:
#                 # \epsilon=0.0 向右下方大幅度拉开
#                 x_offset, y_offset = 3.5, -0.006
#                 txt = r'Lossless ($\epsilon$=0.0)' 
#                 arrow_props = dict(arrowstyle="-", color="gray", lw=1.0, alpha=0.7)
#             # elif rate == 0.001:
#             #     # \epsilon=0.001 向右拉平
#             #     x_offset, y_offset = 5.0, -0.001
#             #     arrow_props = dict(arrowstyle="-", color="gray", lw=1.0, alpha=0.7)
#             # elif rate == 0.005:
#             #     # \epsilon=0.005 向右上方大幅拉开
#             #     x_offset, y_offset = 2.5, 0.005
#                 arrow_props = dict(arrowstyle="-", color="gray", lw=1.0, alpha=0.7)
#             elif rate == 0.01:
#                 # \epsilon=0.01 向左上方拉开
#                 x_offset, y_offset = -2.5, 0.005
#                 arrow_props = dict(arrowstyle="-", color="gray", lw=1.0, alpha=0.7)
                
#             # 下面的点由于比较稀疏，继续用近距离的纯文本标注
#             elif rate == 0.02:
#                 x_offset, y_offset = -1.0, -0.004
#             elif rate == 0.03:
#                 x_offset, y_offset = 0.0, 0.004
#             elif rate == 0.04:
#                 x_offset, y_offset = -1.5, -0.004
#             elif rate == 0.05:
#                 x_offset, y_offset = -1.0, 0.003
#             elif rate == 0.1:
#                 x_offset, y_offset = 0.0, -0.004
#             elif rate == 0.12:
#                 x_offset, y_offset = -1.0, 0.003
#             elif rate == 0.15:
#                 x_offset, y_offset = 1.0, -0.004
#             else:
#                 x_offset, y_offset = 0.5, 0.003
            
#             # 统一使用代表算法的蓝色 (#1f77b4)
#             text_color = '#1f77b4'
            
#             # ⭐ 统一放大每个数据点的文字标注，fontsize=16
#             if arrow_props:
#                 ax.annotate(txt, 
#                             xy=(ours_sizes[i], ours_mmlu[i]), 
#                             xytext=(ours_sizes[i] + x_offset, ours_mmlu[i] + y_offset),
#                             fontsize=16, color=text_color, fontweight='bold', ha='center',
#                             arrowprops=arrow_props)
#             else:
#                 ax.annotate(txt, 
#                             xy=(ours_sizes[i], ours_mmlu[i]), 
#                             xytext=(ours_sizes[i] + x_offset, ours_mmlu[i] + y_offset),
#                             fontsize=16, color=text_color, fontweight='bold', ha='center')

#     # 基线文本提示微调 (⭐ 字号放大至 16)
#     ax.annotate('GPTQ', (gptq_size, gptq_mmlu), xytext=(gptq_size + 1.5, gptq_mmlu - 0.0015),
#                 fontsize=16, color="#59ac59", fontweight='bold')
#     ax.annotate('AWQ', (awq_size, awq_mmlu), xytext=(awq_size + 1.5, awq_mmlu + 0.001),
#                 fontsize=16, color="#f8a154", fontweight='bold')
    
#     ax.annotate('zlib/gzip', (gzip_size, gzip_mmlu), xytext=(gzip_size, gzip_mmlu + 0.002),
#                 fontsize=16, color='#F7CF49', fontweight='bold', ha='center')
#     ax.annotate('zstd', (zstd_size, zstd_mmlu), xytext=(zstd_size, zstd_mmlu - 0.0035),
#                 fontsize=16, color='#cc98fd', fontweight='bold', ha='center')

#     # 7. 图表细节美化
#     ax.set_xlabel('Compression Ratio  (%)')
#     ax.set_ylabel('MMLU Accuracy')
#     # ax.set_title('Compression Ratio vs. MMLU Accuracy on Llama-3.1-8B')

#     ax.grid(True, linestyle='--', alpha=0.6)
#     ax.set_xlim(0.0, 82.0) 
#     ax.set_ylim(0.630, 0.690)

#     # 图例设置
#     ax.legend(loc='lower right', frameon=True, shadow=False, edgecolor='black')

#     # 8. 保存并展示
#     output_filename = 'pareto_frontier_llama3_1_8b_epsilon.pdf'
#     plt.savefig(output_filename, format='pdf', bbox_inches='tight')
#     print(f"图表已成功生成并保存为: {output_filename}")

# if __name__ == "__main__":
#     main()


# import json
# import matplotlib.pyplot as plt

# def main():
#     # 1. 读取 JSON 数据 (您的压缩方法结果)
#     json_file_path = 'parsed_metrics.json'
#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#     except FileNotFoundError:
#         print(f"Error: 找不到文件 {json_file_path}，请确保它与此脚本在同一目录下。")
#         return

#     # 从 JSON 提取所需的数据
#     ours_sizes = [item['Size (%)'] for item in data]
#     ours_mmlu = [item['mmlu/acc'] for item in data]
#     lossy_rates = [item['Rate'] for item in data]

#     # 2. 硬编码基线模型数据
#     gptq_size = 25.0
#     gptq_mmlu = 0.6402

#     awq_size = 25.0
#     awq_mmlu = 0.6665

#     # 3. 全局样式设置
#     plt.rcParams.update({
#         'font.size': 12,
#         'axes.labelsize': 14,
#         'axes.titlesize': 14,
#         'xtick.labelsize': 12,
#         'ytick.labelsize': 12,
#         'legend.fontsize': 12,
#         'font.family': 'serif', 
#     })

#     fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

#     # 4. 绘制核心数据点与曲线
#     # 先画主体蓝色折线，图例中的 r 已替换为 \epsilon
#     ax.plot(ours_sizes, ours_mmlu, marker='o', markersize=6, linestyle='-', linewidth=2.5, 
#             color='#1f77b4', label=r'Ours (Variable $\epsilon$)', zorder=3)
            
#     # 高亮修饰：将 s=40 (与折线的 markersize=6 视觉等大)，保留精巧的黑色边框
#     ax.scatter(ours_sizes[0], ours_mmlu[0], color='#1f77b4', marker='o', s=40, 
#                edgecolors='black', linewidths=1.2, label=r'Ours (Lossless, $\epsilon$=0.0)', zorder=5)

#     # 绘制量化基线散点图
#     ax.scatter(gptq_size, gptq_mmlu, marker='^', s=150, color='#2ca02c', label='GPTQ 4-bit', zorder=4)
#     ax.scatter(awq_size, awq_mmlu, marker='*', s=200, color='#ff7f0e', label='AWQ 4-bit', zorder=4)

#     # 5. 添加标注 (引线标注防重叠系统)
#     key_rates = [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15]
    
#     for i, rate in enumerate(lossy_rates):
#         if rate in key_rates:
#             txt = f'$\\epsilon$={rate}'
            
#             # 使用引线属性
#             arrow_props = None
            
#             if rate == 0.0:
#                 # \epsilon=0.0 向右下方大幅度拉开
#                 x_offset, y_offset = 3.5, -0.006
#                 txt = r'Lossless ($\epsilon$=0.0)' 
#                 arrow_props = dict(arrowstyle="-", color="gray", lw=0.8, alpha=0.7)
#             elif rate == 0.001:
#                 # \epsilon=0.001 向右拉平
#                 x_offset, y_offset = 5.0, -0.001
#                 arrow_props = dict(arrowstyle="-", color="gray", lw=0.8, alpha=0.7)
#             elif rate == 0.005:
#                 # \epsilon=0.005 向右上方大幅拉开
#                 x_offset, y_offset = 2.5, 0.005
#                 arrow_props = dict(arrowstyle="-", color="gray", lw=0.8, alpha=0.7)
#             elif rate == 0.01:
#                 # \epsilon=0.01 向左上方拉开
#                 x_offset, y_offset = -2.5, 0.005
#                 arrow_props = dict(arrowstyle="-", color="gray", lw=0.8, alpha=0.7)
                
#             # 下面的点由于比较稀疏，继续用近距离的纯文本标注
#             elif rate == 0.02:
#                 x_offset, y_offset = -1.0, -0.004
#             elif rate == 0.03:
#                 x_offset, y_offset = 0.0, 0.004
#             elif rate == 0.04:
#                 x_offset, y_offset = -1.5, -0.004
#             elif rate == 0.05:
#                 x_offset, y_offset = -1.0, 0.003
#             elif rate == 0.1:
#                 x_offset, y_offset = 0.0, -0.004
#             elif rate == 0.12:
#                 x_offset, y_offset = -1.0, 0.003
#             elif rate == 0.15:
#                 x_offset, y_offset = 1.0, -0.004
#             else:
#                 x_offset, y_offset = 0.5, 0.003
            
#             # 统一使用代表算法的蓝色 (#1f77b4)
#             text_color = '#1f77b4'
            
#             if arrow_props:
#                 ax.annotate(txt, 
#                             xy=(ours_sizes[i], ours_mmlu[i]), 
#                             xytext=(ours_sizes[i] + x_offset, ours_mmlu[i] + y_offset),
#                             fontsize=11, color=text_color, fontweight='bold', ha='center',
#                             arrowprops=arrow_props)
#             else:
#                 ax.annotate(txt, 
#                             xy=(ours_sizes[i], ours_mmlu[i]), 
#                             xytext=(ours_sizes[i] + x_offset, ours_mmlu[i] + y_offset),
#                             fontsize=11, color=text_color, fontweight='bold', ha='center')

#     # 基线文本提示微调
#     ax.annotate('GPTQ', (gptq_size, gptq_mmlu), xytext=(gptq_size + 1.2, gptq_mmlu - 0.0015),
#                 fontsize=11, color='#2ca02c', fontweight='bold')
#     ax.annotate('AWQ', (awq_size, awq_mmlu), xytext=(awq_size + 1.2, awq_mmlu + 0.001),
#                 fontsize=11, color='#ff7f0e', fontweight='bold')

#     # 6. 图表细节美化
#     ax.set_xlabel('Model Size (%)')
#     ax.set_ylabel('MMLU Zero-shot Accuracy')
#     ax.set_title('Compression Ratio vs. MMLU Accuracy on Llama-3.1-8B')

#     ax.grid(True, linestyle='--', alpha=0.6)
    
#     # ⭐ 核心恢复：由于移除了 78% 的 zlib/zstd，将 X 轴范围恢复到 0-55 左右，使曲线重新居中放大
#     ax.set_xlim(0.0, 55.0) 
#     ax.set_ylim(0.630, 0.690)

#     # 图例设置
#     ax.legend(loc='lower right', frameon=True, shadow=False, edgecolor='black')

#     # 7. 保存并展示
#     output_filename = 'pareto_frontier_llama3_1_8b_no_baselines.pdf'
#     plt.savefig(output_filename, format='pdf', bbox_inches='tight')
#     print(f"图表已成功生成并保存为: {output_filename}")

# if __name__ == "__main__":
#     main()