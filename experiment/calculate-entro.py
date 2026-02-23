import re
import os

def process_delta_log(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    # 读取日志文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 清除由于文本复制或读取产生的 标识
    content_clean = re.sub(r'\\', '', content)
    
    # 2. 将文本展平，消除换行符带来的断行截断问题，确保正则可以顺利匹配
    content_flat = content_clean.replace('\n', ' ')

    # 3. 使用全局正则匹配提取所有数据，容忍多余空格
    sizes_str = re.findall(r'Size:\s*([\d.]+)%', content_flat)
    comps_str = re.findall(r'Comp:\s*([\d.]+)\s*MB/s', content_flat)
    decomps_str = re.findall(r'Decomp:\s*([\d.]+)\s*MB/s', content_flat)

    # 转换为浮点数
    sizes = [float(x) for x in sizes_str]
    comps = [float(x) for x in comps_str]
    decomps = [float(x) for x in decomps_str]

    # 打印调试信息，确认抓取到的数量
    print(f"[调试信息] 共找到 Size: {len(sizes)} 个, Comp: {len(comps)} 个, Decomp: {len(decomps)} 个\n")

    # 确保至少成功提取了10组数据
    if len(sizes) < 10 or len(comps) < 10 or len(decomps) < 10:
        print("未能从文件中解析出完整的10组有效数据，请检查日志格式。")
        return

    print("================ 实验结果汇总 ================")

    # ---------- 前5个模型统计 ----------
    size_top5 = sizes[:5]
    comp_top5 = comps[:5]
    decomp_top5 = decomps[:5]
    
    # 压缩率公式：包括未压缩的原始大小 (即1个100%)，总共有 1 + 5 = 6 项
    avg_size_5 = (100.0 + sum(size_top5)) / 6
    avg_comp_5 = sum(comp_top5) / 5
    avg_decomp_5 = sum(decomp_top5) / 5

    size_str_5 = " + ".join([f"{v:.2f}%" for v in size_top5])
    comp_str_5 = " + ".join([f"{v:.2f}" for v in comp_top5])
    decomp_str_5 = " + ".join([f"{v:.2f}" for v in decomp_top5])

    print("【前 5 个模型】")
    print(f"  压缩率: (100.00% + {size_str_5}) / 6 = {avg_size_5:.2f}%")
    print(f"  压缩 Throughput: ({comp_str_5}) / 5 = {avg_comp_5:.2f} MB/s")
    print(f"  解压缩 Throughput: ({decomp_str_5}) / 5 = {avg_decomp_5:.2f} MB/s\n")

    # ---------- 前10个模型统计 ----------
    size_top10 = sizes[:10]
    comp_top10 = comps[:10]
    decomp_top10 = decomps[:10]
    
    # 压缩率公式：包括未压缩的原始大小 (即1个100%)，总共有 1 + 10 = 11 项
    avg_size_10 = (100.0 + sum(size_top10)) / 11
    avg_comp_10 = sum(comp_top10) / 10
    avg_decomp_10 = sum(decomp_top10) / 10

    size_str_10 = " + ".join([f"{v:.2f}%" for v in size_top10])
    comp_str_10 = " + ".join([f"{v:.2f}" for v in comp_top10])
    decomp_str_10 = " + ".join([f"{v:.2f}" for v in decomp_top10])

    print("【前 10 个模型】")
    print(f"  压缩率: (100.00% + {size_str_10}) / 11 = {avg_size_10:.2f}%")
    print(f"  压缩 Throughput: ({comp_str_10}) / 10 = {avg_comp_10:.2f} MB/s")
    print(f"  解压缩 Throughput: ({decomp_str_10}) / 10 = {avg_decomp_10:.2f} MB/s\n")

if __name__ == "__main__":
    # 请确保您的文件命名为 enrtodelta-bert_large_uncased.log，并将其与此脚本放在同级目录
    process_delta_log("Llama-3.1-70B/entrodelta-Llama-3.1-70B.log")