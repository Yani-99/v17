import re

def process_log(file_path):
    # 读取原始日志文件
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 清除由于文本复制或读取产生的 标识，并将全文本平铺，处理因换行导致的数据截断问题
    content_clean = re.sub(r'\\', '', content)
    content_flat = content_clean.replace('\n', ' ')

    # 正则表达式匹配所需字段，依次提取: 方法名, Comp%, C-Speed, D-Speed
    pattern = r'(zlib|gzip|zstd)\s+\|\s+[\d.]+\s+\|\s+([\d.]+)%\s+\|\s+([\d.]+)\s+\|\s*([\d.]+)'
    matches = re.findall(pattern, content_flat)

    # 初始化存储结构
    methods = ['zlib', 'gzip', 'zstd']
    data = {m: {'comp': [], 'cspeed': [], 'dspeed': []} for m in methods}

    # 数据解析入库
    for match in matches:
        method_name, comp_str, cspeed_str, dspeed_str = match
        data[method_name]['comp'].append(float(comp_str))
        data[method_name]['cspeed'].append(float(cspeed_str))
        data[method_name]['dspeed'].append(float(dspeed_str))

    # 输出计算表达式和结果
    for method in methods:
        print(f"================ {method.upper()} 汇总 ================")
        
        comp_data = data[method]['comp']
        cspeed_data = data[method]['cspeed']
        dspeed_data = data[method]['dspeed']

        # ---------- 前5个模型统计 ----------
        comp_top5 = comp_data[:5]
        cspeed_top5 = cspeed_data[:5]
        dspeed_top5 = dspeed_data[:5]
        
        # 压缩率公式：包括未压缩的原始大小 (即1个100%)，总共有 1 + 5 = 6 项
        avg_comp_5 = (100.0 + sum(comp_top5)) / 6
        avg_cspeed_5 = sum(cspeed_top5) / 5
        avg_dspeed_5 = sum(dspeed_top5) / 5

        comp_str_5 = " + ".join([f"{v:.2f}%" for v in comp_top5])
        cspeed_str_5 = " + ".join([f"{v:.2f}" for v in cspeed_top5])
        dspeed_str_5 = " + ".join([f"{v:.2f}" for v in dspeed_top5])

        print("【前 5 个模型】")
        print(f"  压缩率: (100.00% + {comp_str_5}) / 6 = {avg_comp_5:.2f}%")
        print(f"  C-Speed: ({cspeed_str_5}) / 5 = {avg_cspeed_5:.2f} MB/s")
        print(f"  D-Speed: ({dspeed_str_5}) / 5 = {avg_dspeed_5:.2f} MB/s\n")

        # ---------- 前10个模型统计 ----------
        comp_top10 = comp_data[:10]
        cspeed_top10 = cspeed_data[:10]
        dspeed_top10 = dspeed_data[:10]
        
        # 压缩率公式：包括未压缩的原始大小 (即1个100%)，总共有 1 + 10 = 11 项
        avg_comp_10 = (100.0 + sum(comp_top10)) / 11
        avg_cspeed_10 = sum(cspeed_top10) / 10
        avg_dspeed_10 = sum(dspeed_top10) / 10

        comp_str_10 = " + ".join([f"{v:.2f}%" for v in comp_top10])
        cspeed_str_10 = " + ".join([f"{v:.2f}" for v in cspeed_top10])
        dspeed_str_10 = " + ".join([f"{v:.2f}" for v in dspeed_top10])

        print("【前 10 个模型】")
        print(f"  压缩率: (100.00% + {comp_str_10}) / 11 = {avg_comp_10:.2f}%")
        print(f"  C-Speed: ({cspeed_str_10}) / 10 = {avg_cspeed_10:.2f} MB/s")
        print(f"  D-Speed: ({dspeed_str_10}) / 10 = {avg_dspeed_10:.2f} MB/s\n")

if __name__ == "__main__":
    process_log("Llama-3.1-70B/fake-tra.log")
    # process_log("Llama-3.1-8B/traditional-llama3.1-8b.log")