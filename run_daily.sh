#!/bin/bash
# A股每日选股定时任务
# 每天收盘后自动执行

cd /root/.openclaw/workspace/a-stock-quant-strategies
source venv/bin/activate

# 获取当前日期和时间
DATE=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$DATE] 开始执行每日选股..."

# 运行选股程序（不带 --train，使用已有模型）
python daily_stock_picks.py 2>&1 | tee -a logs/daily_$(date +%Y%m%d).log

# 推送完成通知到 Kimi
python3 notify_kimi.py

echo "[$DATE] 选股完成，报告已生成"
