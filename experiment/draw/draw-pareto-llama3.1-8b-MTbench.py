import json
import matplotlib.pyplot as plt


def main():
    # 1. 读取 JSON 数据 (您的压缩方法结果)
    
    json_file_path = 'parsed_metrics-llama-3.1-8b.json'
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: 找不到文件 {json_file_path}，请确保它与此脚本在同一目录下。")
        return

    # 从 JSON 提取所需的数据
    ours_sizes = [item['Size (%)'] for item in data]
    ours_mtscore = [8.248,8.3924,8.2644,8.2611,8.1699,8.2631,8.2931,8.2335,7.7937,6.8521,5.5335]
    lossy_rates = [0,0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.1,0.12,0.15]

    # 2. 硬编码基线模型数据
    # LLama-3.1-8b
    gptq_size = 25.0
    gptq_mtscore = 8.1032

    awq_size = 25.0
    awq_mtscore = 8.2166

    gzip_size = 78.94
    gzip_mtscore = 8.248

    zlib_size = 78.94
    zlib_mtscore = 8.248

    # fm_delta_size = 60.74
    # fm_delta_mtscore = 0.6812

    zstd_size = 77.54
    zstd_mtscore = 8.248

    bitdelta_size = 7
    bitdelta_mtscore = 7.5

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
    ax.plot([ours_sizes[0],  zstd_size, gzip_size],
            [ours_mtscore[0],  zstd_mtscore, gzip_mtscore],
            linestyle='--', color='gray', linewidth=1.5, alpha=0.5, zorder=1)
    
    # ⭐ 修复 3：由于 MT-Bench 的跨度比 GSM8K 大得多，将偏移量按比例放大至 0.05
    ax.annotate('Lossless Compression\n Baseline', xy=(60, gzip_mtscore), xytext=(60, gzip_mtscore + 0.05),
                fontsize=16, color='dimgray', style='italic', ha='center')

    # 5. 绘制核心数据点与曲线
    ax.plot(ours_sizes, ours_mtscore, marker='o', markersize=8, linestyle='-', linewidth=3.0, 
            color="#62a2d3", label=r'EntroDelta (Variable $\epsilon$)', zorder=3)
            
    # 无损点标记尺寸等比放大
    ax.scatter(ours_sizes[0], ours_mtscore[0], color='#62a2d3', marker='o', s=80, 
               edgecolors='black', linewidths=1.5, label=r'EntroDelta (Lossless, $\epsilon$=0.0)', zorder=5)

    # 绘制其他基线散点图
    ax.scatter(gptq_size, gptq_mtscore, marker='^', s=200, color="#59ac59", label='GPTQ 4-bit', zorder=4)
    ax.scatter(awq_size, awq_mtscore, marker='*', s=250, color="#f8a154", label='AWQ 4-bit', zorder=4)
    ax.scatter(gzip_size, gzip_mtscore, marker='X', s=150, color='#F7CF49', label='zlib/gzip (Lossless)', zorder=4)
    ax.scatter(zstd_size, zstd_mtscore, marker='P', s=150, color='#cc98fd', label='zstd (Lossless)', zorder=4)
    ax.scatter(bitdelta_size, bitdelta_mtscore, marker='s', s=150, color='#e49191', label='BitDelta', zorder=4)

    # 6. 添加标注 
    key_rates = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15]

    for i, rate in enumerate(lossy_rates):
        if rate in key_rates:
            txt = f'$\\epsilon$={rate}'
            
            # 使用 offset points 绝对像素偏移，不受 Y 轴大跨度拉伸的影响
            if rate == 0.0:
                x_offset, y_offset = 75, -20
                txt = r'Lossless ($\epsilon$=0.0)' 
            elif rate == 0.01:
                x_offset, y_offset = 0, 20
            elif rate == 0.02:
                x_offset, y_offset = 18, -20
            elif rate == 0.03:
                x_offset, y_offset = -8, -25
            elif rate == 0.04:
                x_offset, y_offset = 7, 25
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
                        xy=(ours_sizes[i], ours_mtscore[i]), 
                        xytext=(x_offset, y_offset),
                        textcoords='offset points',
                        fontsize=16, color=text_color, fontweight='bold', 
                        ha='center', va='center')

    # 基线文本提示微调 (同样按比例放大了数据偏移量)
    ax.annotate('GPTQ', (gptq_size, gptq_mtscore), xytext=(gptq_size + 2.2, gptq_mtscore - 0.25),
                fontsize=16, color="#59ac59", fontweight='bold',ha='right')
    
    # 保持 AWQ 在左侧贴合
    ax.annotate('AWQ', (awq_size, awq_mtscore), xytext=(awq_size - 1.5, awq_mtscore - 0.05),
                fontsize=16, color="#f8a154", fontweight='bold', ha='right')
    
    ax.annotate('zlib/gzip', (gzip_size, gzip_mtscore), xytext=(gzip_size-1, gzip_mtscore + 0.12),
                fontsize=16, color='#F7CF49', fontweight='bold', ha='center')
    ax.annotate('zstd', (zstd_size, zstd_mtscore), xytext=(zstd_size-0.5, zstd_mtscore - 0.25),
                fontsize=16, color='#cc98fd', fontweight='bold', ha='center')
    ax.annotate('BitDelta', (bitdelta_size, bitdelta_mtscore), xytext=(bitdelta_size-6 , bitdelta_mtscore-0.25),
                fontsize=16, color='#e49191', fontweight='bold', ha='left')

    # 7. 图表细节美化
    ax.set_xlabel('Compression Ratio (%)')
    # Y 轴标签通常不用 "mtscore Accuracy"，而是 "MT-Bench Score"，更加专业
    ax.set_ylabel('MT-Bench Score') 

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0.0, 82.0) 
    
    # ⭐ 修复 2：MT-Bench 的分数范围通常在 5 到 9 之间，必须放开 Y 轴限制！
    ax.set_ylim(5.0, 8.8)

    # 图例设置 
    ax.legend(loc='lower right', frameon=True, shadow=False, edgecolor='black')

    # 8. 保存并展示
    output_filename = 'pareto_frontier_llama3_1_8b_mtscore.pdf'
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"图表已成功生成并保存为: {output_filename}")

if __name__ == "__main__":
    main()