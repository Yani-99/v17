#!/bin/bash

# ==========================================
# 配置区域
# ==========================================
# 您想要测试的线程数列表，可以根据论文要求自由增减
# THREADS=(1 2 4 8 16 32)
THREADS=(48)

# 负载阈值：设定为多少以下认为服务器是“空闲”的？
# 既然您的服务器是 48 核，设定为 3 或 4 是比较严谨的空闲标准
IDLE_THRESHOLD=3
cd ..

# 统一存放测试结果和日志的文件
LOG_FILE="benchmark_throughput.log"

echo "==================================================="
echo "自动化 Benchmarking 脚本已启动"
echo "将在系统负载低于 $IDLE_THRESHOLD 时自动执行测试"
echo "==================================================="

# ==========================================
# 监控与执行主循环
# ==========================================
while true; do
    # 直接读取 /proc/loadavg 获取最近 1 分钟的负载均值，并提取整数部分
    load_int=$(cat /proc/loadavg | awk -F. '{print $1}')
    
    # 检查负载是否满足空闲条件
    if [ "$load_int" -lt "$IDLE_THRESHOLD" ]; then
        echo "[$(date)] 触发测试！当前系统负载 < $IDLE_THRESHOLD，环境纯净。" | tee -a "$LOG_FILE"
        
        # 开始遍历每个线程配置进行测试
        for t in "${THREADS[@]}"; do
            echo "" | tee -a "$LOG_FILE"
            echo "---------------------------------------------------" | tee -a "$LOG_FILE"
            echo "[$(date)] 开始测试 --> 线程数: $t" | tee -a "$LOG_FILE"
            
            # 【关键优化】：计算绑核范围
            # 为了防止操作系统在满打满算 48 个逻辑核之间随机乱弹线程（增加上下文切换延迟），
            # 严格将进程绑定在 0 到 t-1 号核心上。
            core_limit=$((t - 1))
            
            # 打印将要执行的具体命令
            echo "执行命令: taskset -c 0-$core_limit python bench/main.py --num $t" | tee -a "$LOG_FILE"
            
            # 正式执行您的 Python 脚本，并将标准输出和错误流全部追加到日志文件中
            taskset -c 0-$core_limit python bench/main.py --num $t >> "$LOG_FILE" 2>&1
            
            echo "[$(date)] 线程数 $t 的测试已结束。" | tee -a "$LOG_FILE"
            
            # 每次跑完一个配置，强制休息 10 秒钟，让操作系统回收内存和清理 Cache
            sleep 10
        done
        
        echo "===================================================" | tee -a "$LOG_FILE"
        echo "[$(date)] 所有配置已全部测试完毕，完美收工！" | tee -a "$LOG_FILE"
        break  # 测试完成，跳出死循环，脚本结束
        
    else
        # 如果依然拥挤，获取完整的 1分钟、5分钟、15分钟 负载数据打印出来看看
        full_load=$(cat /proc/loadavg | awk '{print $1, $2, $3}')
        echo "[$(date)] 当前服务器较拥挤 (负载: $full_load)，挂起等待 15 分钟..."
        
        # 睡眠 900 秒（15 分钟）后再次检查
        sleep 900
    fi
done