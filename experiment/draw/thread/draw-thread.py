import re
import matplotlib.pyplot as plt
import numpy as np
import os

# 配置日志文件路径
LOG_FILE = "benchmark_throughput.log"

def parse_log(filepath):
    threads = []
    c_speeds_mb = []
    d_speeds_mb = []
    
    current_thread = None
    
    if not os.path.exists(filepath):
        print(f"[错误] 找不到日志文件: {filepath}")
        return [], [], []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            # 1. 匹配线程数声明行：[Sat 07 Mar 2026 11:27:18 PM EST] 开始测试 --> 线程数: 1
            m_thread = re.search(r'开始测试 --> 线程数:\s*(\d+)', line)
            if m_thread:
                current_thread = int(m_thread.group(1))
            
            # 2. 匹配 Rate = 0.0 的结果行
            # 格式例如: 0.0      | 45.24%     | 331.4 MB/s   | 352.0 MB/s   | N/A (No Eval)
            m_rate = re.match(r'^0\.0\s+\|\s+[\d\.]+%?\s+\|\s+([\d\.]+)\s+MB/s\s+\|\s+([\d\.]+)\s+MB/s', line.strip())
            
            if m_rate and current_thread is not None:
                c_speed = float(m_rate.group(1))
                d_speed = float(m_rate.group(2))
                
                # 记录数据
                threads.append(current_thread)
                c_speeds_mb.append(c_speed)
                d_speeds_mb.append(d_speed)
                
                # 重置当前线程，防止重复读取（等待下一个线程块）
                current_thread = None

    return threads, c_speeds_mb, d_speeds_mb

def plot_charts(threads, c_speeds_mb, d_speeds_mb):
    if not threads:
        print("[警告] 没有解析到任何数据，请检查日志文件内容。")
        return
        
    print(f"成功解析数据: 线程数={threads}")
    
    # 转换为 GB/s (1 GB = 1024 MB) 以适应屋顶图/高吞吐量展示
    c_speeds_gb = [x / 1024.0 for x in c_speeds_mb]
    d_speeds_gb = [x / 1024.0 for x in d_speeds_mb]
    
    # 计算加速比 (Speedup)
    base_c = c_speeds_mb[0]
    base_d = d_speeds_mb[0]
    c_speedups = [x / base_c for x in c_speeds_mb]
    d_speedups = [x / base_d for x in d_speeds_mb]
    
    # # ==========================================
    # # 图表 1: 绝对吞吐量双折线图 (Throughput)
    # # ==========================================
    # plt.figure(figsize=(8, 4))
    
    # plt.plot(threads, d_speeds_gb, marker='o', markersize=8, color='#1f77b4', linewidth=2.5, label='Decompression (D-Speed)')
    # plt.plot(threads, c_speeds_gb, marker='s', markersize=8, color='#d62728', linewidth=2.5, label='Compression (C-Speed)')
    
    # # 坐标轴设置
    # plt.xscale('log', base=2)
    # plt.xticks(threads, labels=[str(t) for t in threads], fontsize=11)
    # plt.yticks(fontsize=11)
    # plt.xlabel('Number of Threads', fontsize=13, fontweight='bold')
    # plt.ylabel('Throughput (GB/s)', fontsize=13, fontweight='bold')
    # plt.title('EntroDelta Multi-threading Throughput', fontsize=15, fontweight='bold')
    
    # plt.grid(True, which="major", linestyle="--", alpha=0.6)
    # plt.legend(fontsize=12, loc='upper left')
    
    # plt.tight_layout()
    # # plt.savefig('fig_throughput_scaling.pdf', format='pdf', bbox_inches='tight')
    # # plt.savefig('fig_throughput_scaling.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
    # ==========================================
    # 图表 2: 加速比双折线图 (Speedup)
    # ==========================================
    plt.figure(figsize=(8, 4))
    
    plt.plot(threads, d_speedups, marker='o', markersize=8, color='#62a2d3', linewidth=2.5, label='Decompression Speedup')
    plt.plot(threads, c_speedups, marker='s', markersize=8, color='#59ac59', linewidth=2.5, label='Compression Speedup')
    
    # 加入理想线性加速作为基准线 (Ideal Linear Scaling)
    plt.plot(threads, threads, linestyle='--', color='gray', linewidth=2, label='Ideal Linear Speedup')
    
    plt.xscale('log', base=2)
    # y 轴使用线性还是对数系取决于您想强调什么，这里使用线性，但可开启格线
    plt.xticks(threads, labels=[str(t) for t in threads], fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0, max(threads) + 2) # 设置Y轴上限稍微高出理想线一点
    
    plt.xlabel('Number of Threads', fontsize=15, fontweight='bold')
    plt.ylabel('Speedup (x-times)', fontsize=15, fontweight='bold')
    # plt.title('EntroDelta Strong Scalability (Speedup)', fontsize=15, fontweight='bold')
    
    plt.grid(True, which="major", linestyle="--", alpha=0.6)
    plt.legend(fontsize=15, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('speedup_scaling.pdf', format='pdf', bbox_inches='tight')
    # plt.savefig('fig_speedup_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()

    # print("图表生成完毕！已保存为:")
    # print("- 吞吐量图: fig_throughput_scaling.png / .pdf")
    # print("- 加速比图: fig_speedup_scaling.png / .pdf")

if __name__ == "__main__":
    t, c_mb, d_mb = parse_log(LOG_FILE)
    plot_charts(t, c_mb, d_mb)