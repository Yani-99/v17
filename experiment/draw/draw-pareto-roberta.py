import json
import matplotlib.pyplot as plt

def main():
    # 1. 读取 JSON 数据 (您的压缩方法结果)
    json_file_path = 'parsed_metrics-roberta-large-1.json'
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: 找不到文件 {json_file_path}，请确保它与此脚本在同一目录下。")
        return

    # 从 JSON 提取所需的数据
    ours_sizes = [item['Size (%)'] for item in data]
    ours_accuracy = [item['eval_accuracy'] for item in data]
    lossy_rates = [item['Rate'] for item in data]

    # 2. 硬编码基线模型数据
    gzip_size = 58.70
    gzip_accuracy = 0.9060

    zlib_size = 58.70
    zlib_accuracy = 0.9060

    fm_delta_size = 92.53
    fm_delta_accuracy = 0.9060

    zstd_size = 59.04
    zstd_accuracy = 0.9060

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
    ax.plot([ours_sizes[0], gzip_size, zstd_size, fm_delta_size],
            [ours_accuracy[0], gzip_accuracy, zstd_accuracy, fm_delta_accuracy],
            linestyle='--', color='gray', linewidth=1.5, alpha=0.5, zorder=1)
    
    # ⭐ 将基线说明文字整体下移 (从 45 调整为 20，贴近虚线)
    ax.annotate('Lossless Compression\n Baseline', xy=(70, gzip_accuracy), xytext=(25, -35),
                textcoords='offset points',
                fontsize=16, color='dimgray', style='italic', ha='center')

    # 5. 绘制核心数据点与曲线
    ax.plot(ours_sizes, ours_accuracy, marker='o', markersize=8, linestyle='-', linewidth=3.0, 
            color="#62a2d3", label=r'EntroDelta (Variable $\epsilon$)', zorder=3)
            
    # 无损点标记尺寸等比放大
    ax.scatter(ours_sizes[0], ours_accuracy[0], color='#62a2d3', marker='o', s=80, 
               edgecolors='black', linewidths=1.5, label=r'EntroDelta (Lossless, $\epsilon$=0.0)', zorder=5)

    # 绘制其他基线散点图 
    ax.scatter(gzip_size, gzip_accuracy, marker='X', s=150, color='#F7CF49', label='zlib/gzip (Lossless)', zorder=4)
    ax.scatter(zstd_size, zstd_accuracy, marker='P', s=150, color='#cc98fd', label='zstd (Lossless)', zorder=4)
    ax.scatter(fm_delta_size, fm_delta_accuracy, marker='D', s=100, color="#C4877B", label='FMdelta', zorder=4)

    # 6. 添加标注 
    key_rates = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15]
    
    # ⭐ 统一定义灰色引导线样式
    arrow_props_gray = dict(arrowstyle='-', color='gray', alpha=0.7, lw=1)

    for i, rate in enumerate(lossy_rates):
        if rate in key_rates:
            txt = f'$\\epsilon$={rate}'
            arrow_props = None
            
            # 针对不同点进行偏移微调
            if rate == 0.0:
                x_offset, y_offset = 95, -35     
                txt = r'Lossless ($\epsilon$=0.0)' 
                arrow_props = arrow_props_gray
            elif rate == 0.01:
                x_offset, y_offset = 40, 35      # ⭐ 0.01 往右移 (x_offset 从 15 改为 40)
                arrow_props = arrow_props_gray
            elif rate == 0.02:
                x_offset, y_offset = 15, -45     
                arrow_props = arrow_props_gray
            elif rate == 0.03:
                x_offset, y_offset = 0, 40       # ⭐ 0.03 (原笔误0.13) 下来一点 (y_offset 从 70 改为 50)
                arrow_props = arrow_props_gray
            elif rate == 0.04:
                x_offset, y_offset = -10, -65    
                arrow_props = arrow_props_gray
            elif rate == 0.05:
                x_offset, y_offset = -35, 35     
                arrow_props = arrow_props_gray
            elif rate == 0.1:
                x_offset, y_offset = 35, 0       # 0.1 维持原状
            elif rate == 0.12:
                x_offset, y_offset = 60, 30      # ⭐ 0.12 往右移，加上灰色引导线
                arrow_props = arrow_props_gray
            elif rate == 0.15:
                x_offset, y_offset = 35, -25     # ⭐ 0.15 加上灰色引导线
                arrow_props = arrow_props_gray
            else:
                x_offset, y_offset = 15, 15
            
            text_color = '#1f77b4'
            
            ax.annotate(txt, 
                        xy=(ours_sizes[i], ours_accuracy[i]), 
                        xytext=(x_offset, y_offset),
                        textcoords='offset points',
                        fontsize=16, color=text_color, fontweight='bold', 
                        ha='center', va='center',
                        arrowprops=arrow_props) 

    # 基线文字标注 
    ax.annotate('zlib/gzip', xy=(gzip_size, gzip_accuracy), xytext=(20, 15),
                textcoords='offset points',
                fontsize=16, color='#F7CF49', fontweight='bold', ha='center')
    
    ax.annotate('zstd', xy=(zstd_size, zstd_accuracy), xytext=(-10, -25),
                textcoords='offset points',
                fontsize=16, color='#cc98fd', fontweight='bold', ha='center')
                
    ax.annotate('FMdelta', xy=(fm_delta_size, fm_delta_accuracy), xytext=(0, 20),
                textcoords='offset points',
                fontsize=16, color='#C4877B', fontweight='bold', ha='center')

    # 7. 图表细节美化
    ax.set_xlabel('Compression Ratio (%)')
    ax.set_ylabel('Accuracy')

    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.set_xlim(0.0, 100.0) 
    
    # 动态计算 Y 轴范围
    min_accuracy = min(min(ours_accuracy), gzip_accuracy)
    max_accuracy = max(max(ours_accuracy), gzip_accuracy)
    y_padding = (max_accuracy - min_accuracy) * 0.1 if max_accuracy != min_accuracy else 0.1
    
    ax.set_ylim(min_accuracy - y_padding * 1.5, max_accuracy + y_padding * 2.0)

    # 图例设置
    ax.legend(loc='lower right', frameon=True, shadow=False, edgecolor='black')

    # 8. 保存并展示
    output_filename = 'pareto_frontier_roberta-accuracy.pdf'
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"图表已成功生成并保存为: {output_filename}")

if __name__ == "__main__":
    main()

# import json
# import matplotlib.pyplot as plt

# def main():
#     # 1. 读取 JSON 数据 (您的压缩方法结果)
#     json_file_path = 'parsed_metrics-roberta-large-1.json'
#     try:
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#     except FileNotFoundError:
#         print(f"Error: 找不到文件 {json_file_path}，请确保它与此脚本在同一目录下。")
#         return

#     # 从 JSON 提取所需的数据
#     ours_sizes = [item['Size (%)'] for item in data]
#     ours_accuracy = [item['eval_accuracy'] for item in data]
#     lossy_rates = [item['Rate'] for item in data]

#     # 2. 硬编码基线模型数据
#     gzip_size = 58.70
#     gzip_accuracy = 0.9060

#     zlib_size = 58.70
#     zlib_accuracy = 0.9060

#     fm_delta_size = 92.53
#     fm_delta_accuracy = 0.9060

#     zstd_size = 59.04
#     zstd_accuracy = 0.9060

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
#     ax.plot([ours_sizes[0], gzip_size, zstd_size, fm_delta_size],
#             [ours_accuracy[0], gzip_accuracy, zstd_accuracy, fm_delta_accuracy],
#             linestyle='--', color='gray', linewidth=1.5, alpha=0.5, zorder=1)
    
#     # ⭐ 将基线说明文字大幅拔高，彻底避开下方的数据点和标签
#     ax.annotate('Lossless Compression\n Baseline', xy=(70, gzip_accuracy), xytext=(0, 45),
#                 textcoords='offset points',
#                 fontsize=16, color='dimgray', style='italic', ha='center')

#     # 5. 绘制核心数据点与曲线
#     ax.plot(ours_sizes, ours_accuracy, marker='o', markersize=8, linestyle='-', linewidth=3.0, 
#             color="#62a2d3", label=r'EntroDelta (Variable $\epsilon$)', zorder=3)
            
#     # 无损点标记尺寸等比放大
#     ax.scatter(ours_sizes[0], ours_accuracy[0], color='#62a2d3', marker='o', s=80, 
#                edgecolors='black', linewidths=1.5, label=r'EntroDelta (Lossless, $\epsilon$=0.0)', zorder=5)

#     # 绘制其他基线散点图 
#     ax.scatter(gzip_size, gzip_accuracy, marker='X', s=150, color='#F7CF49', label='zlib/gzip (Lossless)', zorder=4)
#     ax.scatter(zstd_size, zstd_accuracy, marker='P', s=150, color='#cc98fd', label='zstd (Lossless)', zorder=4)
#     ax.scatter(fm_delta_size, fm_delta_accuracy, marker='D', s=100, color='#8c564b', label='FMdelta', zorder=4)

#     # 6. 添加标注 (⭐ 核心修复：多级拉链式排版，应对极端拥挤)
#     key_rates = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15]

#     for i, rate in enumerate(lossy_rates):
#         if rate in key_rates:
#             txt = f'$\\epsilon$={rate}'
            
#             # 使用 offset points 绝对像素偏移，针对 roberta 的平坦且密集曲线进行极致高低分层
#             if rate == 0.0:
#                 x_offset, y_offset = 60, -25     # 无损点字长，向右下方远远推开
#                 txt = r'Lossless ($\epsilon$=0.0)' 
#             elif rate == 0.01:
#                 x_offset, y_offset = 0, 20       # 第一层：正上方
#             elif rate == 0.02:
#                 x_offset, y_offset = 0, -25      # 第一层：正下方
#             elif rate == 0.03:
#                 x_offset, y_offset = 0, 45       # ⭐ 第二层：大幅拔高，越过 0.01
#             elif rate == 0.04:
#                 x_offset, y_offset = 0, -50      # ⭐ 第二层：大幅下压，躲开 0.02
#             elif rate == 0.05:
#                 x_offset, y_offset = -15, 20     # 回到第一层，稍微偏左上
#             elif rate == 0.1:
#                 x_offset, y_offset = 30, 0       # 0.1 在中间下坠段，放正右侧最安全
#             elif rate == 0.12:
#                 x_offset, y_offset = -20, 25     # 放左上方
#             elif rate == 0.15:
#                 x_offset, y_offset = 25, -25     # 放右下方
#             else:
#                 x_offset, y_offset = 15, 15
            
#             text_color = '#1f77b4'
            
#             # 无引线，纯靠精准绝对像素偏隔离
#             ax.annotate(txt, 
#                         xy=(ours_sizes[i], ours_accuracy[i]), 
#                         xytext=(x_offset, y_offset),
#                         textcoords='offset points',
#                         fontsize=16, color=text_color, fontweight='bold', 
#                         ha='center', va='center')

#     # 基线文字标注 (大幅拉开 zlib 和 zstd 的距离)
#     ax.annotate('zlib/gzip', xy=(gzip_size, gzip_accuracy), xytext=(20, 25),
#                 textcoords='offset points',
#                 fontsize=16, color='#F7CF49', fontweight='bold', ha='center')
    
#     ax.annotate('zstd', xy=(zstd_size, zstd_accuracy), xytext=(-20, -25),
#                 textcoords='offset points',
#                 fontsize=16, color='#cc98fd', fontweight='bold', ha='center')
                
#     ax.annotate('FMdelta', xy=(fm_delta_size, fm_delta_accuracy), xytext=(0, 20),
#                 textcoords='offset points',
#                 fontsize=16, color='#8c564b', fontweight='bold', ha='center')

#     # 7. 图表细节美化
#     ax.set_xlabel('Compression Ratio (%)')
#     ax.set_ylabel('Accuracy')

#     ax.grid(True, linestyle='--', alpha=0.6)
    
#     ax.set_xlim(0.0, 100.0) 
    
#     # 动态计算 Y 轴范围，为我们高高拔起的 0.03 (y_offset=45) 留足顶部空间
#     min_accuracy = min(min(ours_accuracy), gzip_accuracy)
#     max_accuracy = max(max(ours_accuracy), gzip_accuracy)
#     y_padding = (max_accuracy - min_accuracy) * 0.1 if max_accuracy != min_accuracy else 0.1
#     # 强制给顶部多加一点 Padding，防止文字顶到天花板
#     ax.set_ylim(min_accuracy - y_padding, max_accuracy + y_padding * 1.5)

#     # 图例设置
#     ax.legend(loc='lower right', frameon=True, shadow=False, edgecolor='black')

#     # 8. 保存并展示
#     output_filename = 'pareto_frontier_roberta-accuracy_zipper.pdf'
#     plt.savefig(output_filename, format='pdf', bbox_inches='tight')
#     print(f"图表已成功生成并保存为: {output_filename}")

# if __name__ == "__main__":
#     main()