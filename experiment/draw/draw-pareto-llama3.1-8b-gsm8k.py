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
    ours_gsm8k = [item['gsm8k/acc'] for item in data]
    lossy_rates = [item['Rate'] for item in data]

    # 2. 硬编码基线模型数据
    # LLama-3.1-8b
    gptq_size = 25.0
    gptq_gsm8k = 0.6702

    awq_size = 25.0
    awq_gsm8k = 0.7293

    gzip_size = 78.94
    gzip_gsm8k = 0.753601213

    zlib_size = 78.94
    zlib_gsm8k = 0.753601213

    # fm_delta_size = 60.74
    # fm_delta_gsm8k = 0.6812

    bitdelta_size = 7
    bitdelta_gsm8k = 0.73

    zstd_size = 77.54
    zstd_gsm8k = 0.753601213

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
    ax.plot([ours_sizes[0],  zstd_size, gzip_size],
            [ours_gsm8k[0],  zstd_gsm8k, gzip_gsm8k],
            linestyle='--', color='gray', linewidth=1.5, alpha=0.5, zorder=1)
    
    # ⭐ 修复 1：将 Y 轴偏移量从 0.008 降到 0.003，让文字更贴近虚线
    ax.annotate('Lossless Compression\n Baseline', xy=(60, gzip_gsm8k), xytext=(60, gzip_gsm8k + 0.003),
                fontsize=16, color='dimgray', style='italic', ha='center')

    # 5. 绘制核心数据点与曲线
    ax.plot(ours_sizes, ours_gsm8k, marker='o', markersize=8, linestyle='-', linewidth=3.0, 
            color="#62a2d3", label=r'EntroDelta (Variable $\epsilon$)', zorder=3)
            
    # 无损点标记尺寸等比放大
    ax.scatter(ours_sizes[0], ours_gsm8k[0], color='#62a2d3', marker='o', s=80, 
               edgecolors='black', linewidths=1.5, label=r'EntroDelta (Lossless, $\epsilon$=0.0)', zorder=5)

    # 绘制其他基线散点图
    ax.scatter(gptq_size, gptq_gsm8k, marker='^', s=200, color="#59ac59", label='GPTQ 4-bit', zorder=4)
    ax.scatter(awq_size, awq_gsm8k, marker='*', s=250, color="#f8a154", label='AWQ 4-bit', zorder=4)
    ax.scatter(gzip_size, gzip_gsm8k, marker='X', s=150, color='#F7CF49', label='Zlib/Gzip (Lossless)', zorder=4)
    ax.scatter(zstd_size, zstd_gsm8k, marker='P', s=150, color='#cc98fd', label='Zstd (Lossless)', zorder=4)
    ax.scatter(bitdelta_size, bitdelta_gsm8k, marker='s', s=150, color='#e49191', label='BitDelta', zorder=4)

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
                x_offset, y_offset = 20, -20
            elif rate == 0.03:
                x_offset, y_offset = 0, -25
            elif rate == 0.04:
                x_offset, y_offset = 5, 28
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
                        xy=(ours_sizes[i], ours_gsm8k[i]), 
                        xytext=(x_offset, y_offset),
                        textcoords='offset points',
                        fontsize=16, color=text_color, fontweight='bold', 
                        ha='center', va='center')

    # 基线文本提示微调
    ax.annotate('GPTQ', (gptq_size, gptq_gsm8k), xytext=(gptq_size + 1.5, gptq_gsm8k - 0.006),
                fontsize=16, color="#59ac59", fontweight='bold')
    # ⭐ 修复 2：将 AWQ 移到左边，通过减去 X 轴坐标并设置 ha='right' 实现完美左侧贴合
    ax.annotate('AWQ', (awq_size, awq_gsm8k), xytext=(awq_size - 1.5, awq_gsm8k  - 0.005),
                fontsize=16, color="#f8a154", fontweight='bold', ha='right')
    
    ax.annotate('Zlib/Gzip', (gzip_size, gzip_gsm8k), xytext=(gzip_size-1, gzip_gsm8k + 0.008),
                fontsize=16, color='#F7CF49', fontweight='bold', ha='center')
    ax.annotate('Zstd', (zstd_size, zstd_gsm8k), xytext=(zstd_size-0.5, zstd_gsm8k - 0.02),
                fontsize=16, color='#cc98fd', fontweight='bold', ha='center')
    ax.annotate('BitDelta', (bitdelta_size, bitdelta_gsm8k), xytext=(bitdelta_size -6, bitdelta_gsm8k-0.017),
                fontsize=16, color='#e49191', fontweight='bold', ha='left')

    # 7. 图表细节美化
    ax.set_xlabel('Compression Ratio  (%)')
    ax.set_ylabel('GSM8K Accuracy') 

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0.0, 82.0) 
    
    ax.set_ylim(0.530, 0.780)

    # 图例设置 
    ax.legend(loc='lower right', frameon=True, shadow=False, edgecolor='black')

    # 8. 保存并展示
    output_filename = 'pareto_frontier_llama3_1_8b_gsm8k.pdf'
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"图表已成功生成并保存为: {output_filename}")

if __name__ == "__main__":
    main()