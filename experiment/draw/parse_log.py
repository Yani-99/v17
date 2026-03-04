import re
import json
import pandas as pd

def parse_llm_harness_log(file_path):
    """
    解析大型语言模型评估日志，提取 Rate, Model, Size, Speed 和 Metrics
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        log_content = f.read()

    # 定义正则表达式匹配模式
    # 匹配头部信息如： [========== Rate: 0.005 = Model: llama3.1-8b-1==========]
    rate_model_pattern = r"\[========== Rate: ([\d\.]+) = Model: (.*?)==========\]"
    # 匹配性能信息如： -> Size: 45.01% | Comp: 1226.5 MB/s | Decomp: 5039.2 MB/s |
    perf_pattern = r"-> Size: ([\d\.]+)% \| Comp: ([\d\.]+) MB/s \| Decomp: ([\d\.]+) MB/s \|"
    # 匹配详细的 Metrics 字典如： Metrics: {'mmlu/acc': 0.68..., ...}
    metrics_pattern = r"Metrics: (\{.*?\})"

    # 使用 Rate 作为分隔符将整个日志分割成不同的评估阶段
    # 注意，这样做会把原始字符串切分，我们需要保留或重新拼接 Rate 头信息
    sections = re.split(r"\[========== Rate: ", log_content)
    
    parsed_data = []

    # 第一部分通常是 Baseline 准备或加载的信息，实际的 Rate 循环从 sections[1] 开始
    for section in sections[1:]:
        # 由于我们使用 [========== Rate: 作为分隔符，切分后需要把这个前缀补回来以确保正则匹配正常工作
        section_text = "[========== Rate: " + section
        
        # 搜索关键信息
        rate_model_match = re.search(rate_model_pattern, section_text)
        perf_match = re.search(perf_pattern, section_text)
        metrics_match = re.search(metrics_pattern, section_text)
        
        # 确保所有关键信息都匹配到，避免因为某次崩溃或者日志损坏导致读取报错
        if rate_model_match and perf_match and metrics_match:
            # 1. 提取实验设置参数
            rate = float(rate_model_match.group(1))
            model = rate_model_match.group(2).strip()
            
            # 2. 提取压缩参数与速度参数
            size_percent = float(perf_match.group(1))
            comp_speed = float(perf_match.group(2))
            decomp_speed = float(perf_match.group(3))
            
            # 3. 解析所有的准确率 / 速度 Metrics 字典
            metrics_str = metrics_match.group(1)
            # Python 日志中打印的字典格式往往是用单引号，我们需要将其替换成双引号来满足 JSON 的解析标准
            metrics_str = metrics_str.replace("'", '"')
            
            try:
                metrics_dict = json.loads(metrics_str)
            except json.JSONDecodeError as e:
                print(f"在 Rate: {rate} 阶段解析 Metrics JSON 失败: {e}")
                metrics_dict = {}
            
            # 4. 组装当前阶段的所有记录
            row_data = {
                "Rate": rate,
                "Model": model,
                "Size (%)": size_percent,
                "Comp Speed (MB/s)": comp_speed,
                "Decomp Speed (MB/s)": decomp_speed
            }
            # 将 metrics_dict 中的每一个 key-value 追加到 row_data 中
            row_data.update(metrics_dict)
            
            parsed_data.append(row_data)

    return parsed_data

if __name__ == "__main__":
    # 日志文件路径
    # log_file = "entro-llama3.1-8.log"
    log_file = "roberta-large-1-eval.log"
    
    # 获取解析后的列表字典
    data = parse_llm_harness_log(log_file)
    
    # 转化为 Pandas DataFrame 对象，方便输出二维表格
    df = pd.DataFrame(data)
    
    # 1. 保存为完整包含层级关系的 JSON 文件
    json_output_file = "parsed_metrics-roberta-large-1.json"
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"数据已成功提取并保存到 JSON: {json_output_file}")
    
    # 2. 保存为包含所有信息的 CSV 表格文件
    # csv_output_file = "parsed_metrics.csv"
    # df.to_csv(csv_output_file, index=False)
    # print(f"数据已成功提取并保存到 CSV: {csv_output_file}")


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
#     # X轴: 压缩后的相对大小 Size (%)
#     # Y轴: MMLU Zero-shot 准确率 mmlu/acc
#     ours_sizes = [item['Size (%)'] for item in data]
#     ours_mmlu = [item['mmlu/acc'] for item in data]
#     lossy_rates = [item['Rate'] for item in data]

#     # 2. 硬编码基线模型数据 (GPTQ 4-bit 和 AWQ 4-bit)
#     # 4-bit 量化相对于 16-bit 原始模型的理论大小约为 25%
#     gptq_size = 25.0
#     gptq_mmlu = 0.6402

#     awq_size = 25.0
#     awq_mmlu = 0.6665

#     # 3. 全局样式设置 (符合顶级学术会议 PDF 矢量图的标准)
#     plt.rcParams.update({
#         'font.size': 12,
#         'axes.labelsize': 14,
#         'axes.titlesize': 14,
#         'xtick.labelsize': 12,
#         'ytick.labelsize': 12,
#         'legend.fontsize': 12,
#         'font.family': 'serif', # 使用学术论文常用的衬线字体
#     })

#     fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

#     # 4. 绘制数据点与曲线
#     # 绘制我们方法的折线图 (带有标记)
#     ax.plot(ours_sizes, ours_mmlu, marker='o', markersize=6, linestyle='-', linewidth=2.5, 
#             color='#1f77b4', label='Ours (Variable Lossy Rate)', zorder=3)

#     # 绘制 GPTQ 和 AWQ 的散点图
#     ax.scatter(gptq_size, gptq_mmlu, marker='^', s=150, color='#2ca02c', label='GPTQ 4-bit', zorder=4)
#     ax.scatter(awq_size, awq_mmlu, marker='*', s=200, color='#ff7f0e', label='AWQ 4-bit', zorder=4)

#     # 5. 添加标注 (Annotations)
#     # 为了防止图表过于拥挤，我们选择几个关键的 Lossy Rate 进行文字标注
#     # 您可以根据视觉效果在 key_rates 列表中增删您想展示的 Rate
#     key_rates = [0.0, 0.01, 0.05, 0.1, 0.15]
    
#     for i, rate in enumerate(lossy_rates):
#         if rate in key_rates:
#             txt = f'r={rate}'
#             # 根据数据点位置动态调整文本偏移量，防止互相遮挡
#             y_offset = 0.003 if rate <= 0.05 else -0.005
#             x_offset = 0.5
            
#             ax.annotate(txt, (ours_sizes[i], ours_mmlu[i]), 
#                         xytext=(ours_sizes[i] + x_offset, ours_mmlu[i] + y_offset),
#                         fontsize=11, color='#1f77b4', fontweight='bold')

#     # 给 GPTQ 和 AWQ 添加文本提示
#     ax.annotate('GPTQ', (gptq_size, gptq_mmlu), xytext=(gptq_size - 1.5, gptq_mmlu - 0.005),
#                 fontsize=11, color='#2ca02c', fontweight='bold')
#     ax.annotate('AWQ', (awq_size, awq_mmlu), xytext=(awq_size - 1.5, awq_mmlu + 0.003),
#                 fontsize=11, color='#ff7f0e', fontweight='bold')

#     # 添加 "Better" 箭头指示优化方向 (左上角为优：体积更小，精度更高)
#     # 基于您的数据范围动态设置箭头位置
#     ax.annotate('Better', xy=(15.0, 0.68), xytext=(20.0, 0.67),
#                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
#                 fontsize=12, fontweight='bold', ha='center')

#     # 6. 图表细节美化
#     ax.set_xlabel('Model Size (%) $\downarrow$')
#     ax.set_ylabel('MMLU Zero-shot Accuracy $\uparrow$')
#     ax.set_title('Compression Ratio vs. MMLU Accuracy on Llama-3.1-8B')

#     # 设置网格线 (仅显示 Y 轴和 X 轴的主要网格，使图表清晰)
#     ax.grid(True, linestyle='--', alpha=0.6)

#     # ⭐ 核心设置：反转 X 轴 (让体积从 100% 降到 0% 从左向右排列，符合"压缩"的直觉)
#     ax.invert_xaxis()

#     # 调整坐标轴范围，留出适当空白
#     ax.set_xlim(50.0, 0.0) 
#     # Y 轴范围覆盖从最差 (0.64左右) 到最好 (0.68+)
#     ax.set_ylim(0.635, 0.69)

#     # 添加图例
#     ax.legend(loc='lower left', frameon=True, shadow=False, edgecolor='black')

#     # 7. 保存并展示
#     output_filename = 'pareto_frontier_llama3_1_8b_real_data.pdf'
#     plt.savefig(output_filename, format='pdf', bbox_inches='tight')
#     print(f"图表已成功生成并保存为: {output_filename}")
    
#     # 如果是在有图形界面的环境中，取消下面这行的注释即可弹窗查看
#     # plt.show()

# if __name__ == "__main__":
#     main()