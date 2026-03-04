import matplotlib.pyplot as plt
import numpy as np

def main():
    # 1. 核心数据配置区 (整合 9 个模型的数据，并将各自独立的基线数据合并到每个模型的字典中)
    models_data = [
        {
            "name": "gpt2",
            "rates": [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15],
            "comp_ratios": [68.68, 68.65, 64.15, 56.69, 42.33, 26.65, 16.00, 10.13, 2.67, 1.91, 1.26],
            "color": "#62a2d3", "marker": "o",
            "baselines": [
                (84.44, "#F7CF49", "zlib/gzip (Lossless)", "X"),
                (84.36, "#cc98fd", "zstd (Lossless)", "P"),
                (66.38, "#75ba8f", "FMdelta (Lossless)", "D")
            ]
        },
        {
            "name": "bert-large-uncased",
            "rates": [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15],
            "comp_ratios": [69.06, 67.69, 63.87, 59.34, 48.61, 39.80, 27.98, 17.92, 1.07, 0.29, 0.04],
            "color": "#62a2d3", "marker": "o",
            "baselines": [
                (92.84, "#F7CF49", "zlib/gzip (Lossless)", "X"),
                (92.77, "#cc98fd", "zstd (Lossless)", "P"),
                (68.23, "#75ba8f", "FMdelta (Lossless)", "D")
            ]
        },
        {
            "name": "roberta-base",
            "rates": [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15],
            "comp_ratios": [50.79, 30.58, 24.40, 19.75, 12.58, 4.99, 1.80, 0.71, 0.17, 0.17, 0.15],
            "color": "#62a2d3", "marker": "o",
            "baselines": [
                (85.86, "#F7CF49", "zlib/gzip (Lossless)", "X"),
                (85.35, "#cc98fd", "zstd (Lossless)", "P"),
                (92.52, "#75ba8f", "FMdelta (Lossless)", "D")
            ]
        },
        {
            "name": "roberta-large",
            "rates": [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15],
            "comp_ratios": [30.40, 30.40, 30.24, 29.55, 27.40, 24.91, 22.36, 19.54, 3.41, 1.53, 0.48],
            "color": "#62a2d3", "marker": "o",
            "baselines": [
                (58.70, "#F7CF49", "zlib/gzip (Lossless)", "X"),
                (59.04, "#cc98fd", "zstd (Lossless)", "P"),
                (92.53, "#75ba8f", "FMdelta (Lossless)", "D")
            ]
        },
        {
            "name": "Llama-2-7b-hf",
            "rates": [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15],
            "comp_ratios": [50.96, 50.95, 50.76, 50.36, 48.48, 46.20, 43.82, 41.30, 26.38, 17.55, 9.09],
            "color": "#62a2d3", "marker": "o",
            "baselines": [
                (76.99, "#F7CF49", "zlib/gzip (Lossless)", "X"),
                (76.56, "#cc98fd", "zstd (Lossless)", "P"),
                (63.96, "#75ba8f", "FMdelta (Lossless)", "D")
            ]
        },
        {
            "name": "Llama-2-13b-hf",
            "rates": [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15],
            "comp_ratios": [50.96, 50.95, 50.75, 50.36, 48.51, 46.26, 43.94, 41.50, 26.85, 17.68, 8.93],
            "color": "#62a2d3", "marker": "o",
            "baselines": [
                (76.95, "#F7CF49", "zlib/gzip (Lossless)", "X"),
                (76.49, "#cc98fd", "zstd (Lossless)", "P"),
                (63.93, "#75ba8f", "FMdelta (Lossless)", "D")
            ]
        },
        {
            "name": "Mistral-7B-v0.1",
            "rates": [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15],
            "comp_ratios": [40.69, 40.68, 40.65, 40.54, 40.16, 39.70, 39.15, 38.57, 35.05, 33.49, 31.14],
            "color": "#62a2d3", "marker": "o",
            "baselines": [
                (79.17, "#F7CF49", "zlib/gzip (Lossless)", "X"),
                (77.84, "#cc98fd", "zstd (Lossless)", "P")
            ]
        },
        {
            "name": "Llama-3.1-8B",
            "rates": [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15],
            "comp_ratios": [45.24, 45.22, 45.01, 44.04, 40.50, 36.53, 32.61, 28.96, 12.07, 6.79, 2.73],
            "color": "#62a2d3", "marker": "o",
            "baselines": [
                (78.94, "#F7CF49", "zlib/gzip (Lossless)", "X"),
                (77.54, "#cc98fd", "zstd (Lossless)", "P")
            ]
        },
        {
            "name": "Llama-3.1-70B",
            "rates": [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15],
            "comp_ratios": [31.67, 31.66, 31.53, 30.97, 28.73, 26.17, 23.36, 20.37, 3.79, 1.63, 0.47],
            "color": "#62a2d3", "marker": "o",
            "baselines": [
                (78.56, "#F7CF49", "zlib/gzip (Lossless)", "X"),
                (76.99, "#cc98fd", "zstd (Lossless)", "P")
            ]
        }
    ]

    # 3. 全局学术排版样式设置 (保持不变)
    plt.rcParams.update({
        'font.size': 13,
        'axes.labelsize': 15,
        'axes.titlesize': 16,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,  # 稍微调小一点以适应子图的图例空间 (由于基线名字较长，这里调成10防止越界)
        'font.family': 'serif',
    })

    # 创建 3x3 的画布，适当放大 figsize 以容纳 9 个模型
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 10), dpi=300)
    
    # 展平 axes 数组，方便用单个循环进行遍历
    axes = axes.flatten()

    # 4 & 5. 遍历 9 个模型并在各个子图中作图
    for i, data in enumerate(models_data):
        ax = axes[i]
        
        # 绘制该子图对应的独立单点基线 (水平虚线 + 起点散点)
        # ⭐ 核心修改点：这里直接从 data["baselines"] 中读取当前模型专属的基线数据
        for comp_ratio, color, label, marker in data["baselines"]:
            ax.axhline(y=comp_ratio, color=color, linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
            ax.scatter(0.0, comp_ratio, color=color, marker=marker, s=80, label=label, zorder=4)

        # 绘制该子图对应的 Ours 曲线
        ax.plot(data["rates"], data["comp_ratios"], 
                marker=data["marker"], markersize=6, linestyle='-', linewidth=2.5, 
                color=data["color"], label="EntroDelta (Ours)", zorder=3)

        # 7. 图表细节与坐标轴美化
        ax.set_title(data["name"])
        
        # 为保持排版清爽，建议仅在最左侧和最下方的子图显示坐标轴文字
        # 但为遵循您“不删改”的要求，此处仍为每个子图保留原有的 label 设置
        ax.set_xlabel(r'Fidelity Tolerance ($\epsilon$)')
        ax.set_ylabel('Compression Ratio (%)')

        # 设置 X 轴的范围与刻度 (0.0 到 0.15) (保持不变)
        ax.set_xlim(-0.005, 0.155)
        ax.set_xticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15])
        
        # ⭐ 核心修改点：统一的 Y 轴高度调整到了 100，以容纳 bert 和 roberta 中高达 92% 的压缩基准线
        ax.set_ylim(0, 100)

        # 开启网格线，透明度调低以免喧宾夺主 (保持不变)
        ax.grid(True, linestyle=':', alpha=0.6)

        # 图例设置 (将图例放在右上角)
        # 为防止遮挡，仅在每个子图内部加上图例
        ax.legend(loc='upper right', frameon=True, shadow=False, edgecolor='black', ncol=1)

    # 自动调整子图间距，防止坐标轴文字与标题重叠
    plt.tight_layout()

    # 8. 导出与保存
    output_filename = 'compression_ratio_9_models.pdf'
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"✅ 图表已成功生成并保存为: {output_filename}")

if __name__ == "__main__":
    main()


# import matplotlib.pyplot as plt
# import numpy as np

# def main():
#     # 1. 核心数据配置区 (整合 9 个模型的数据)
#     # 采用列表嵌套字典的结构，方便后续在 3x3 的网格中遍历画图
#     models_data = [
#         {
#             "name": "gpt2",
#             "rates": [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15],
#             "comp_ratios": [68.68, 68.65, 64.15, 56.69, 42.33, 26.65, 16.00, 10.13, 2.67, 1.91, 1.26],
#             "color": "#62a2d3", "marker": "o"
#         },
#         {
#             "name": "bert-large-uncased",#ber-ner10
#             "rates": [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15],
#             "comp_ratios": [69.06, 67.69, 63.87, 59.34, 48.61, 39.80, 27.98, 17.92, 1.07, 0.29, 0.04],
#             "color": "#62a2d3", "marker": "o"
#         },
#         {
#             "name": "roberta-base",
#             "rates": [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15],
#             "comp_ratios": [50.79, 30.58, 24.40, 19.75, 12.58, 4.99, 1.80, 0.71, 0.17, 0.17, 0.15],
#             "color": "#62a2d3", "marker": "o"
#         },
#         {
#             "name": "roberta-large",
#             "rates": [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15],
#             "comp_ratios": [30.40, 30.40, 30.24, 29.55, 27.40, 24.91, 22.36, 19.54, 3.41, 1.53, 0.48],
#             "color": "#62a2d3", "marker": "o"
#         },
#         {
#             "name": "Llama-2-7b-hf",
#             "rates": [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15],
#             "comp_ratios": [50.96, 50.95, 50.76, 50.36, 48.48, 46.20, 43.82, 41.30, 26.38, 17.55, 9.09],
#             "color": "#62a2d3", "marker": "o"
#         },
#         {
#             "name": "Llama-2-13b-hf",
#             "rates": [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15],
#             "comp_ratios": [50.96, 50.95, 50.75, 50.36, 48.51, 46.26, 43.94, 41.50, 26.85, 17.68, 8.93],
#             "color": "#62a2d3", "marker": "o"
#         },
#         {
#             "name": "Mistral-7B-v0.1",
#             "rates": [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15],
#             "comp_ratios": [40.69, 40.68, 40.65, 40.54, 40.16, 39.70, 39.15, 38.57, 35.05, 33.49, 31.14],
#             "color": "#62a2d3", "marker": "o"
#         },
#         {
#             "name": "Llama-3.1-8B",
#             "rates": [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15],
#             "comp_ratios": [45.24, 45.22, 45.01, 44.04, 40.50, 36.53, 32.61, 28.96, 12.07, 6.79, 2.73],
#             "color": "#62a2d3", "marker": "o"
#         },
#         {
#             "name": "Llama-3.1-70B",
#             "rates": [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.15],
#             "comp_ratios": [31.67, 31.66, 31.53, 30.97, 28.73, 26.17, 23.36, 20.37, 3.79, 1.63, 0.47],
#             "color": "#62a2d3", "marker": "o"
#         }
#     ]

#     # 2. 单点基线数据配置 (保持不变)
#     baselines_gpt2 = [
#         (84.44, "#F7CF49", "zlib/gzip (Lossless)", "X"),
#         (84.36, "#cc98fd", "zstd (Lossless)", "P"),
#         (66.38, "#75ba8f", "FMdelta (Lossless)", "D")
#     ]

#     baselines_bert-large-uncased = [
#         (92.84, "#F7CF49", "zlib/gzip (Lossless)", "X"),
#         (92.77, "#cc98fd", "zstd (Lossless)", "P"),
#         (68.23, "#75ba8f", "FMdelta (Lossless)", "D")
#     ]
#     baselines_roberta-base = [
#         (85.86, "#F7CF49", "zlib/gzip (Lossless)", "X"),
#         (85.35, "#cc98fd", "zstd (Lossless)", "P"),
#         (92.52, "#75ba8f", "FMdelta (Lossless)", "D")
#     ]
#     baselines_roberta-large = [
#         (58.70, "#F7CF49", "zlib/gzip (Lossless)", "X"),
#         (59.04, "#cc98fd", "zstd (Lossless)", "P"),
#         (92.53, "#75ba8f", "FMdelta (Lossless)", "D")
#     ]
#     baselines_Llama_2_7b_hf = [
#         (76.99, "#F7CF49", "zlib/gzip (Lossless)", "X"),
#         (76.56, "#cc98fd", "zstd (Lossless)", "P"),
#         (63.96, "#75ba8f", "FMdelta (Lossless)", "D")
#     ]

#     baselines_Llama_2_13b_hf = [
#         (76.95, "#F7CF49", "zlib/gzip (Lossless)", "X"),
#         (76.49, "#cc98fd", "zstd (Lossless)", "P"),
#         (63.93, "#75ba8f", "FMdelta (Lossless)", "D")
#     ]
#     baselines_Mistral_7B_v0-1 = [
#         (79.17, "#F7CF49", "zlib/gzip (Lossless)", "X"),
#         (77.84, "#cc98fd", "zstd (Lossless)", "P")
#     ]
#     baselines_Llama_3_1_8B = [
#         (78.94, "#F7CF49", "zlib/gzip (Lossless)", "X"),
#         (77.54, "#cc98fd", "zstd (Lossless)", "P")
#     ]
#     baselines_Llama_3_1_70B = [
#         (78.56, "#F7CF49", "zlib/gzip (Lossless)", "X"),
#         (76.99, "#cc98fd", "zstd (Lossless)", "P")
#     ]
#     # 3. 全局学术排版样式设置 (保持不变)
#     plt.rcParams.update({
#         'font.size': 13,
#         'axes.labelsize': 15,
#         'axes.titlesize': 16,
#         'xtick.labelsize': 13,
#         'ytick.labelsize': 13,
#         'legend.fontsize': 12,  # 稍微调小一点以适应子图的图例空间
#         'font.family': 'serif',
#     })

#     # 创建 3x3 的画布，适当放大 figsize 以容纳 9 个模型
#     fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 10), dpi=300)
    
#     # 展平 axes 数组，方便用单个循环进行遍历
#     axes = axes.flatten()

#     # 4 & 5. 遍历 9 个模型并在各个子图中作图
#     for i, data in enumerate(models_data):
#         ax = axes[i]
        
#         # 绘制该子图的单点基线 (水平虚线 + 起点散点)
#         for comp_ratio, color, label, marker in baselines:
#             ax.axhline(y=comp_ratio, color=color, linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
#             ax.scatter(0.0, comp_ratio, color=color, marker=marker, s=80, label=label, zorder=4)

#         # 绘制该子图对应的 Ours 曲线
#         ax.plot(data["rates"], data["comp_ratios"], 
#                 marker=data["marker"], markersize=6, linestyle='-', linewidth=2.5, 
#                 color=data["color"], label="EntroDelta (Ours)", zorder=3)

#         # 7. 图表细节与坐标轴美化
#         ax.set_title(data["name"])
        
#         # 为保持排版清爽，建议仅在最左侧和最下方的子图显示坐标轴文字
#         # 但为遵循您“不删改”的要求，此处仍为每个子图保留原有的 label 设置
#         ax.set_xlabel(r'Fidelity Tolerance ($\epsilon$)')
#         ax.set_ylabel('Compression Ratio (%)')

#         # 设置 X 轴的范围与刻度 (0.0 到 0.15) (保持不变)
#         ax.set_xlim(-0.005, 0.155)
#         ax.set_xticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15])
        
#         # 设置 Y 轴范围 (从 0 到 85) (保持不变，统一的Y轴有助于跨图直观对比模型表现差异)
#         ax.set_ylim(0, 85)

#         # 开启网格线，透明度调低以免喧宾夺主 (保持不变)
#         ax.grid(True, linestyle=':', alpha=0.6)

#         # 图例设置 (将图例放在右上角)
#         # 为防止遮挡，仅在每个子图内部加上图例
#         ax.legend(loc='upper right', frameon=True, shadow=False, edgecolor='black', ncol=1)

#     # 自动调整子图间距，防止坐标轴文字与标题重叠
#     plt.tight_layout()

#     # 8. 导出与保存
#     output_filename = 'compression_ratio_9_models.pdf'
#     plt.savefig(output_filename, format='pdf', bbox_inches='tight')
#     print(f"✅ 图表已成功生成并保存为: {output_filename}")

# if __name__ == "__main__":
#     main()