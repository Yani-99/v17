import re
import ast
import pandas as pd

def extract_metrics_to_csv(log_file_path, csv_file_path):
    # 1. 读取整个日志文件内容
    with open(log_file_path, 'r', encoding='utf-8') as f:
        log_content = f.read()

    # 2. 提取所有的 Rate 值，以便在 CSV 中进行对应
    # 正则匹配形如 "[========== Rate: 0.0001 = Model:" 的部分
    rate_pattern = re.compile(r"\[========== Rate: ([\d.]+) = Model:")
    rates = rate_pattern.findall(log_content)
    
    # 根据文件结构可知，第一次 Baseline Assessment (对应第一个指标) 虽然没有在开头打出 Rate 标签，
    # 但最终总结中它被视为 Rate 0.0，且之后恰好还有12次带 Rate 标签的运行。
    # 为了严谨对齐，如果提取到的 metrics 比 Rate 标签多1，说明第一次为 Baseline (Rate=0.0)
    
    # 3. 提取 Size、Comp、Decomp 以及 Metrics 字典
    # 使用 re.DOTALL 能够兼容日志中间出现的潜在折行或多余空格
    pattern = re.compile(
        r"->\s*Size:\s*([\d.]+)%\s*\|\s*Comp:\s*([\d.]+)\s*MB/s\s*\|\s*Decomp:\s*([\d.]+)\s*MB/s\s*\|\s*Metrics:\s*(\{.*?\})",
        re.DOTALL
    )

    matches = pattern.findall(log_content)

    if not matches:
        print("未能在日志文件中找到相关指标数据！")
        return

    # 对齐 Rate 值列表
    if len(rates) == len(matches) - 1:
        rates = ["0.0"] + rates
    elif len(rates) != len(matches):
        rates = ["Unknown"] * len(matches)

    all_data = []
    
    # 4. 遍历所有匹配结果并组合数据
    for i, match in enumerate(matches):
        size_pct, comp_mbs, decomp_mbs, metrics_str = match
        
        # 将日志可能产生的多余换行符去除，保障 ast.literal_eval 能顺利将其当做单行 dict 解析
        metrics_str = re.sub(r'\s+', ' ', metrics_str)
        
        try:
            # 将类似 "{'mmlu/acc': 0.68...}" 的字符串安全地转换成 Python 字典对象
            metrics_dict = ast.literal_eval(metrics_str)
        except Exception as e:
            print(f"解析第 {i+1} 个 Metrics 字典时发生错误: {e}")
            continue
            
        # 构建当前行的数据基础结构
        row_data = {
            'Rate': rates[i],
            'Size (%)': float(size_pct),
            'Comp (MB/s)': float(comp_mbs),
            'Decomp (MB/s)': float(decomp_mbs)
        }
        
        # 将详细的指标字典数据更新进当前行数据中
        row_data.update(metrics_dict)
        all_data.append(row_data)

    # 5. 使用 pandas 转换为 DataFrame 后直接输出到 CSV (自动处理所有字典的 Key 作为列名)
    df = pd.DataFrame(all_data)
    df.to_csv(csv_file_path, index=False)
    
    print(f"提取完成！成功从 {log_file_path} 提取了 {len(all_data)} 条指标记录。")
    print(f"文件已保存至: {csv_file_path}")

if __name__ == "__main__":
    # 执行提取，你可以根据需要调整输入和输出的文件名
    extract_metrics_to_csv('entro-llama3.1-8.log', 'extracted_metrics.csv')