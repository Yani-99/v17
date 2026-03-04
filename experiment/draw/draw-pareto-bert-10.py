import json
import matplotlib.pyplot as plt

def main():
    # 1. 读取 JSON 数据 (您的压缩方法结果)
    json_file_path = 'parsed_metrics-bert_ner10.json'
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: 找不到文件 {json_file_path}，请确保它与此脚本在同一目录下。")
        return

    # 从 JSON 提取所需的数据
    ours_sizes = [item['Size (%)'] for item in data]
    ours_f1 = [item['eval_f1'] for item in data]
    lossy_rates = [item['Rate'] for item in data]

    # 2. 硬编码基线模型数据
    gzip_size = 78.94
    gzip_f1 = 0.9184

    zlib_size = 78.94
    zlib_f1 = 0.9184

    fm_delta_size = 60.74
    fm_delta_f1 = 0.9184

    zstd_size = 77.54
    zstd_f1 = 0.9184

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
            [ours_f1[0], fm_delta_f1, zstd_f1, gzip_f1],
            linestyle='--', color='gray', linewidth=1.5, alpha=0.5, zorder=1)
    
    # ⭐ 核心修复 1：使用真实的 F1 值 (gzip_f1) 作为锚点，并采用 offset points 绝对偏移，保证永远贴合虚线
    ax.annotate('Lossless Compression\n Baseline', xy=(65, gzip_f1), xytext=(0, 25),
                textcoords='offset points',
                fontsize=16, color='dimgray', style='italic', ha='center')

    # 5. 绘制核心数据点与曲线
    ax.plot(ours_sizes, ours_f1, marker='o', markersize=8, linestyle='-', linewidth=3.0, 
            color="#62a2d3", label=r'EntroDelta (Variable $\epsilon$)', zorder=3)
            
    # 无损点标记尺寸等比放大
    ax.scatter(ours_sizes[0], ours_f1[0], color='#62a2d3', marker='o', s=80, 
               edgecolors='black', linewidths=1.5, label=r'EntroDelta (Lossless, $\epsilon$=0.0)', zorder=5)

    # 绘制其他基线散点图 (已注释掉不相关的 GPTQ/AWQ)
    ax.scatter(gzip_size, gzip_f1, marker='X', s=150, color='#F7CF49', label='zlib/gzip (Lossless)', zorder=4)
    ax.scatter(zstd_size, zstd_f1, marker='P', s=150, color='#cc98fd', label='zstd (Lossless)', zorder=4)
    ax.scatter(fm_delta_size, fm_delta_f1, marker='D', s=100, color='#8c564b', label='FMdelta', zorder=4)

    # 6. 添加标注 (采用极近距离贴身绝对像素偏移)
    key_rates = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15]

    for i, rate in enumerate(lossy_rates):
        if rate in key_rates:
            txt = f'$\\epsilon$={rate}'
            
            # 使用 offset points 绝对像素偏移，距离缩短至 20 像素左右，形成齿轮般紧凑的无重叠排版
            if rate == 0.0:
                x_offset, y_offset = 75, -20
                txt = r'Lossless ($\epsilon$=0.0)' 
            elif rate == 0.01:
                x_offset, y_offset = 0, 20
            elif rate == 0.02:
                x_offset, y_offset = 0, -20
            elif rate == 0.03:
                x_offset, y_offset = -17, 20
            elif rate == 0.04:
                x_offset, y_offset = -15, -20
            elif rate == 0.05:
                x_offset, y_offset = -30, 20
            elif rate == 0.1:
                x_offset, y_offset = 25, -20
            elif rate == 0.12:
                x_offset, y_offset = -25, 20
            elif rate == 0.15:
                x_offset, y_offset = 35, -20
            else:
                x_offset, y_offset = 15, 15
            
            text_color = '#1f77b4'
            
            ax.annotate(txt, 
                        xy=(ours_sizes[i], ours_f1[i]), 
                        xytext=(x_offset, y_offset),
                        textcoords='offset points',
                        fontsize=16, color=text_color, fontweight='bold', 
                        ha='center', va='center')

    # ⭐ 核心修复 2：将基线的文字标注全部切换为绝对磅值偏移 (offset points)，彻底免疫坐标轴数据拉伸
    ax.annotate('zlib/gzip', xy=(gzip_size, gzip_f1), xytext=(0, 20),
                textcoords='offset points',
                fontsize=16, color='#F7CF49', fontweight='bold', ha='center')
    
    ax.annotate('zstd', xy=(zstd_size, zstd_f1), xytext=(0, -20),
                textcoords='offset points',
                fontsize=16, color='#cc98fd', fontweight='bold', ha='center')
                
    ax.annotate('FMdelta', xy=(fm_delta_size, fm_delta_f1), xytext=(0, 20),
                textcoords='offset points',
                fontsize=16, color='#8c564b', fontweight='bold', ha='center')

    # 7. 图表细节美化
    ax.set_xlabel('Compression Ratio (%)')
    ax.set_ylabel('F1 Score')

    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 将 X 轴稍微放宽一点点，容纳大号字体
    ax.set_xlim(0.0, 85.0) 
    
    # ⭐ 核心修复 3：动态计算 Y 轴范围。F1 值可能会跌得很低，用自适应计算最稳妥
    # 寻找数据中的最小值和最大值，并上下留出 10% 的空白缓冲区
    min_f1 = min(min(ours_f1), gzip_f1)
    max_f1 = max(max(ours_f1), gzip_f1)
    y_padding = (max_f1 - min_f1) * 0.1 if max_f1 != min_f1 else 0.1
    ax.set_ylim(min_f1 - y_padding, max_f1 + y_padding)

    # 图例设置
    ax.legend(loc='lower right', frameon=True, shadow=False, edgecolor='black')

    # 8. 保存并展示
    output_filename = 'pareto_frontier_bert-f1.pdf'
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"图表已成功生成并保存为: {output_filename}")

if __name__ == "__main__":
    main()


# import json
# import matplotlib.pyplot as plt

# def main():
#     # 1. 读取 JSON 数据 (您的压缩方法结果)
#     # json_file_path = 'parsed_metrics.json'
#     json_file_path = 'parsed_metrics-bert_ner10.json'
#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#     except FileNotFoundError:
#         print(f"Error: 找不到文件 {json_file_path}，请确保它与此脚本在同一目录下。")
#         return

#     # 从 JSON 提取所需的数据
#     ours_sizes = [item['Size (%)'] for item in data]
#     ours_f1 = [item['eval_f1'] for item in data]
#     lossy_rates = [item['Rate'] for item in data]

#     # 2. 硬编码基线模型数据

#     gzip_size = 78.94
#     gzip_f1 = 0.9184

#     zlib_size = 78.94
#     zlib_f1 = 0.9184

#     fm_delta_size = 60.74
#     fm_delta_f1 = 0.9184

#     zstd_size = 77.54
#     zstd_f1 = 0.9184

#     # 3. 全局样式设置 (已全面放大字号以适应 Paper 缩放)
#     plt.rcParams.update({
#         'font.size': 18,           
#         'axes.labelsize': 22,      
#         'axes.titlesize': 22,      
#         'xtick.labelsize': 18,     
#         'ytick.labelsize': 18,     
#         'legend.fontsize': 16,     
#         'font.family': 'serif', 
#     })

#     fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

#     # 4. 绘制“断层连接线” (Visual Bridge)
#     ax.plot([ours_sizes[0], fm_delta_size, zstd_size, gzip_size],
#             [ours_f1[0], fm_delta_f1, zstd_f1, gzip_f1],
#             linestyle='--', color='gray', linewidth=1.5, alpha=0.5, zorder=1)
    
#     # ⭐ 核心修复 1：将 Baseline 文字大幅向左移动至 X=55 的空白安全区，彻底避开 zlib/gzip
#     ax.annotate('Lossless Compression\n Baseline', xy=(60, 0.6812), xytext=(61, 0.6835),
#                 fontsize=16, color='dimgray', style='italic', ha='center')

#     # 5. 绘制核心数据点与曲线
#     ax.plot(ours_sizes, ours_f1, marker='o', markersize=8, linestyle='-', linewidth=3.0, 
#             color="#62a2d3", label=r'EntroDelta (Variable $\epsilon$)', zorder=3)
            
#     # 无损点标记尺寸等比放大
#     ax.scatter(ours_sizes[0], ours_f1[0], color='#62a2d3', marker='o', s=80, 
#                edgecolors='black', linewidths=1.5, label=r'EntroDelta (Lossless, $\epsilon$=0.0)', zorder=5)

#     # 绘制其他基线散点图
#     # ax.scatter(gptq_size, gptq_f1, marker='^', s=200, color="#59ac59", label='GPTQ 4-bit', zorder=4)
#     # ax.scatter(awq_size, awq_f1, marker='*', s=250, color="#f8a154", label='AWQ 4-bit', zorder=4)
#     ax.scatter(gzip_size, gzip_f1, marker='X', s=150, color='#F7CF49', label='zlib/gzip (Lossless)', zorder=4)
#     ax.scatter(zstd_size, zstd_f1, marker='P', s=150, color='#cc98fd', label='zstd (Lossless)', zorder=4)
#     ax.scatter(fm_delta_size, fm_delta_f1, marker='D', s=100, color='#8c564b', label='FMdelta', zorder=4)

#     # 6. 添加标注 (⭐ 核心修复 2：彻底移除灰线，采用极近距离贴身绝对像素偏移)
#     key_rates = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15]

#     for i, rate in enumerate(lossy_rates):
#         if rate in key_rates:
#             txt = f'$\\epsilon$={rate}'
            
#             # 使用 offset points 绝对像素偏移，距离缩短至 20 像素左右，形成齿轮般紧凑的无重叠排版
#             if rate == 0.0:
#                 # 文字较长，向右多偏移一点，向下避开折线
#                 x_offset, y_offset = 75, -20
#                 txt = r'Lossless ($\epsilon$=0.0)' 
#             elif rate == 0.01:
#                 # 紧贴正上方
#                 x_offset, y_offset = 0, 20
#             elif rate == 0.02:
#                 # 紧贴正下方
#                 x_offset, y_offset = 0, -20
#             elif rate == 0.03:
#                 # 紧贴左上方
#                 x_offset, y_offset = -17, 20
#             elif rate == 0.04:
#                 # 紧贴左下方
#                 x_offset, y_offset = -15, -20
#             elif rate == 0.05:
#                 # 紧贴左上方
#                 x_offset, y_offset = -30, 20
#             elif rate == 0.1:
#                 # 紧贴右下方
#                 x_offset, y_offset = 25, -20
#             elif rate == 0.12:
#                 # 紧贴左上方
#                 x_offset, y_offset = -25, 20
#             elif rate == 0.15:
#                 # 紧贴右下方
#                 x_offset, y_offset = 35, -20
#             else:
#                 x_offset, y_offset = 15, 15
            
#             text_color = '#1f77b4'
            
#             # ⭐ 彻底移除了 arrowprops 逻辑，只保留纯文本标注，并用 offset points 锁死间距
#             ax.annotate(txt, 
#                         xy=(ours_sizes[i], ours_f1[i]), 
#                         xytext=(x_offset, y_offset),
#                         textcoords='offset points',
#                         fontsize=16, color=text_color, fontweight='bold', 
#                         ha='center', va='center')

#     # 基线文本提示微调
#     # ax.annotate('GPTQ', (gptq_size, gptq_f1), xytext=(gptq_size + 1.5, gptq_f1 - 0.0015),
#     #             fontsize=16, color="#59ac59", fontweight='bold')
#     # ax.annotate('AWQ', (awq_size, awq_f1), xytext=(awq_size + 1.5, awq_f1 + 0.001),
#     #             fontsize=16, color="#f8a154", fontweight='bold')
    
#     ax.annotate('zlib/gzip', (gzip_size, gzip_f1), xytext=(gzip_size-1, gzip_f1 + 0.002),
#                 fontsize=16, color='#F7CF49', fontweight='bold', ha='center')
#     ax.annotate('zstd', (zstd_size, zstd_f1), xytext=(zstd_size-0.5, zstd_f1 - 0.0035),
#                 fontsize=16, color='#cc98fd', fontweight='bold', ha='center')

#     # 7. 图表细节美化
#     ax.set_xlabel('Compression Ratio  (%)')
#     ax.set_ylabel('F1 Score')

#     ax.grid(True, linestyle='--', alpha=0.6)
#     ax.set_xlim(0.0, 82.0) 
    
#     # 因为去掉了高高拔起的引线，我们可以稍微将天花板调平一些，让曲线更加居中饱满
#     ax.set_ylim(0.630, 0.692)

#     # 图例设置
#     ax.legend(loc='lower right', frameon=True, shadow=False, edgecolor='black')

#     # 8. 保存并展示
#     output_filename = 'pareto_frontier_bert-f1.pdf'
#     plt.savefig(output_filename, format='pdf', bbox_inches='tight')
#     print(f"图表已成功生成并保存为: {output_filename}")

# if __name__ == "__main__":
#     main()
