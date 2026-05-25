#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股趋势跟踪策略回测（真实数据版）
=================================
数据源: 腾讯财经真实K线
策略: 双均线交叉 / 动量 / MACD
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.api import TencentHistoryAPI
from utils.backtest import BacktestEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / 'data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

STOCK_POOL = {
    'sh600519': '贵州茅台', 'sz000858': '五粮液', 'sh601318': '中国平安',
    'sz000333': '美的集团', 'sh600036': '招商银行',
}


def fetch_data():
    """获取最近2年数据。"""
    end = datetime.now()
    start = end - timedelta(days=730)
    start_str, end_str = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
    return TencentHistoryAPI.get_multi_kline(
        list(STOCK_POOL.keys()), start_str, end_str,
        period='day', adjust='qfq', delay=0.3
    )


# ---------- 策略实现 ----------

def ma_cross_strategy(df, fast=5, slow=20):
    data = df.copy()
    data[f'ma{fast}'] = data['close'].rolling(fast).mean()
    data[f'ma{slow}'] = data['close'].rolling(slow).mean()
    data['signal'] = np.where(data[f'ma{fast}'] > data[f'ma{slow}'], 1, -1)
    data['return'] = data['close'].pct_change()
    data['position'] = data['signal'].shift(1)
    data['strategy_return'] = data['position'] * data['return']
    return data


def momentum_strategy(df, window=20):
    data = df.copy()
    data['momentum'] = data['close'].pct_change(window)
    data['mom_ma'] = data['momentum'].rolling(10).mean()
    data['signal'] = np.where(data['mom_ma'] > 0, 1, -1)
    data['return'] = data['close'].pct_change()
    data['position'] = data['signal'].shift(1)
    data['strategy_return'] = data['position'] * data['return']
    return data


def macd_strategy(df, fast=12, slow=26, signal=9):
    data = df.copy()
    ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
    data['dif'] = ema_fast - ema_slow
    data['dea'] = data['dif'].ewm(span=signal, adjust=False).mean()
    data['signal'] = np.where(data['dif'] > data['dea'], 1, -1)
    data['return'] = data['close'].pct_change()
    data['position'] = data['signal'].shift(1)
    data['strategy_return'] = data['position'] * data['return']
    return data


def backtest_single_stock(df, strategy_name):
    """单只股票策略回测，返回每日收益序列。"""
    df = df.dropna(subset=['strategy_return']).copy()
    if len(df) < 20:
        return None
    df['cum'] = (1 + df['strategy_return']).cumprod()
    total = df['cum'].iloc[-1] - 1
    n_days = len(df)
    n_years = n_days / 242
    annual = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = df['strategy_return'].std() * np.sqrt(242)
    sharpe = (annual - 0.03) / vol if vol > 0 else 0
    dd = (df['cum'] - df['cum'].cummax()) / df['cum'].cummax()
    max_dd = dd.min()
    return {
        'strategy_name': strategy_name,
        'total_return': round(total * 100, 2),
        'annual_return': round(annual * 100, 2),
        'sharpe': round(sharpe, 2),
        'max_drawdown': round(max_dd * 100, 2),
        'n_days': n_days,
    }


def plot_trend_results(all_results, filename):
    """绘制趋势策略对比图。"""
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 按股票聚合展示累计收益
    ax1 = axes[0, 0]
    colors = {'双均线交叉': '#E74C3C', '动量策略': '#3498DB', 'MACD策略': '#2ECC71'}
    for sname, sdata in all_results.items():
        cum_returns = []
        for stock, df in sdata.items():
            cum = (1 + df['strategy_return'].fillna(0)).cumprod()
            cum_returns.append(cum.values)
        if cum_returns:
            min_len = min(len(x) for x in cum_returns)
            avg_cum = np.mean([x[:min_len] for x in cum_returns], axis=0)
            ax1.plot(range(min_len), avg_cum, label=sname, color=colors.get(sname, 'gray'), linewidth=1.5)

    ax1.set_title('平均累计收益对比', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 指标表格
    ax2 = axes[0, 1]
    ax2.axis('off')

    # 回撤
    ax3 = axes[1, 0]
    for sname, sdata in all_results.items():
        cum_returns = []
        for stock, df in sdata.items():
            cum = (1 + df['strategy_return'].fillna(0)).cumprod()
            cum_returns.append(cum.values)
        if cum_returns:
            min_len = min(len(x) for x in cum_returns)
            avg_cum = np.mean([x[:min_len] for x in cum_returns], axis=0)
            dd = (avg_cum - np.maximum.accumulate(avg_cum)) / np.maximum.accumulate(avg_cum) * 100
            ax3.fill_between(range(min_len), dd, 0, alpha=0.3, label=sname)
    ax3.set_title('平均回撤', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 日收益分布
    ax4 = axes[1, 1]
    for sname, sdata in all_results.items():
        rets = []
        for stock, df in sdata.items():
            rets.extend(df['strategy_return'].dropna().values)
        ax4.hist(np.array(rets) * 100, bins=30, alpha=0.4, label=sname)
    ax4.axvline(x=0, color='red', linestyle='--')
    ax4.set_title('日收益分布', fontsize=12, fontweight='bold')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"图表: {filename}")


def main():
    print("=" * 60)
    print("  A股趋势跟踪策略回测（真实数据版）")
    print("=" * 60)

    kline_dict = fetch_data()

    strategies = {
        '双均线交叉': lambda df: ma_cross_strategy(df, 5, 20),
        '动量策略': lambda df: momentum_strategy(df, 20),
        'MACD策略': lambda df: macd_strategy(df, 12, 26, 9),
    }

    all_results = {}
    all_metrics = {}

    for sname, sfunc in strategies.items():
        print(f"\n策略: {sname}")
        sdata = {}
        smetrics = []
        for stock, df in kline_dict.items():
            strat_df = sfunc(df)
            sdata[stock] = strat_df
            metrics = backtest_single_stock(strat_df, sname)
            if metrics:
                smetrics.append(metrics)
                print(f"  {stock}: 累计 {metrics['total_return']:+.2f}%  夏普 {metrics['sharpe']:.2f}  "
                      f"最大回撤 {metrics['max_drawdown']:.2f}%")
        all_results[sname] = sdata
        if smetrics:
            avg = {
                'strategy_name': sname,
                'total_return': round(np.mean([m['total_return'] for m in smetrics]), 2),
                'annual_return': round(np.mean([m['annual_return'] for m in smetrics]), 2),
                'sharpe': round(np.mean([m['sharpe'] for m in smetrics]), 2),
                'max_drawdown': round(np.mean([m['max_drawdown'] for m in smetrics]), 2),
            }
            all_metrics[sname] = avg
            print(f"  >> 平均: 累计 {avg['total_return']:+.2f}%  夏普 {avg['sharpe']:.2f}  "
                  f"最大回撤 {avg['max_drawdown']:.2f}%")

    # 绘图
    chart_file = OUTPUT_DIR / 'trend_backtest.png'
    plot_trend_results(all_results, chart_file)

    # 保存
    result = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategies': all_metrics,
    }
    with open(OUTPUT_DIR / 'trend_backtest_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 完成: {OUTPUT_DIR / 'trend_backtest_result.json'}")


if __name__ == '__main__':
    main()
