#!/usr/bin/env python3
"""
选股完成后自动发送通知到 Kimi
"""
import json
import sys
import subprocess
from datetime import datetime
import os

def send_notification(json_path: str):
    if not os.path.exists(json_path):
        msg = f"⚠️ {datetime.now().strftime('%Y-%m-%d')} 选股脚本运行完成，但未生成报告文件，请检查日志。"
        _send(msg)
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
    lines = [
        f"📊 {date} 每日选股完成",
        "",
        "明日 Top 10 推荐：",
    ]

    for i, s in enumerate(data.get('top10', []), 1):
        pred = s.get('pred_return', 0) * 100
        lines.append(
            f"{i}. {s['name']}({s['code']}) 预测:{pred:+.2f}% "
            f"现价:{s['close']} 涨跌:{s['change_pct']:+.2f}% "
            f"PE:{s['pe']} ROE:{s['roe']:.1f}%"
        )

    lines.append("")
    lines.append("⚠️ 仅供学习研究，不构成投资建议。")

    msg = "\n".join(lines)
    _send(msg)

def _send(msg: str):
    # 使用 openclaw CLI 发送消息到 kimi-claw
    cmd = [
        "openclaw", "message", "send",
        "--channel", "kimi-claw",
        "--target", "main",
        "-m", msg,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"[notify] 发送失败: {result.stderr}", file=sys.stderr)
        else:
            print("[notify] 已推送通知到 Kimi")
    except Exception as e:
        print(f"[notify] 发送异常: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 默认路径
        today = datetime.now().strftime('%Y-%m-%d')
        json_path = f"/root/.openclaw/workspace/a-stock-quant-strategies/data/daily_picks/{today}_stock_picks.json"
    else:
        json_path = sys.argv[1]
    send_notification(json_path)
