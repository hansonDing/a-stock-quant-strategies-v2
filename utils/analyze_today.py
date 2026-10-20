#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
今日A股收盘数据分析工具
======================
获取实时行情，输出市场概况、涨跌排序、板块分析。
不涉及选股预测（预测功能已拆分到 daily_stock_picks.py / tomorrow_picks.py）。
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.api import TencentQuoteAPI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / 'data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

STOCKS = {
    'sh600519': '贵州茅台', 'sz000858': '五粮液', 'sh601318': '中国平安',
    'sz000333': '美的集团', 'sh600036': '招商银行', 'sh600276': '恒瑞医药',
    'sh601888': '中国中免', 'sz002594': '比亚迪', 'sh603259': '药明康德',
    'sz300750': '宁德时代', 'sh600030': '中信证券', 'sz000001': '平安银行',
    'sh601012': '隆基绿能', 'sz002415': '海康威视', 'sh600887': '伊利股份',
    'sh601166': '兴业银行', 'sz000568': '泸州老窖', 'sh600309': '万华化学',
    'sz002142': '宁波银行', 'sh603288': '海天味业', 'sz300059': '东方财富',
    'sh601668': '中国建筑', 'sz000002': '万科A', 'sh600900': '长江电力',
    'sz300760': '迈瑞医疗', 'sh601398': '工商银行', 'sz002230': '科大讯飞',
    'sh600028': '中国石化', 'sz000063': '中兴通讯', 'sh601857': '中国石油',
}


def main():
    print("=" * 80)
    print(f"今日A股收盘数据分析 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("=" * 80)

    quotes = TencentQuoteAPI.get_quotes(list(STOCKS.keys()))
    if quotes.empty:
        print("获取行情失败")
        return

    # 类型转换
    numeric_cols = ['close', 'open', 'high', 'low', 'volume', 'amount',
                    'change_pct', 'turnover', 'pe', 'pb', 'change']
    for col in numeric_cols:
        if col in quotes.columns:
            quotes[col] = pd.to_numeric(quotes[col], errors='coerce')

    quotes = quotes[quotes['close'] > 0].copy()
    quotes['name'] = quotes['code_key'].map(STOCKS)
    quotes = quotes.sort_values('change_pct', ascending=False).reset_index(drop=True)

    # 打印
    print(f"\n共获取 {len(quotes)} 只股票")
    print("-" * 80)
    print(f"{'名称':<10}{'代码':<12}{'收盘价':>10}{'涨跌额':>10}{'涨跌幅':>10}{'换手率':>8}{'成交额(万)':>12}{'PE':>8}")
    print("-" * 80)

    total_amount = 0
    for _, row in quotes.iterrows():
        name = str(row.get('name', 'N/A'))[:8]
        code = row.get('code', 'N/A')
        close = row.get('close', 0)
        change = row.get('change', 0)
        pct = row.get('change_pct', 0)
        turnover = row.get('turnover', 0)
        amount = row.get('amount', 0)
        pe = row.get('pe', 0)
        total_amount += amount
        print(f"{name:<10}{code:<12}{close:>10.2f}{change:>+10.2f}{pct:>+9.2f}%{turnover:>7.2f}%{amount:>11.0f}万{pe:>7.2f}")

    print("-" * 80)
    print(f"今日总成交额: {total_amount/10000:.2f} 亿元")

    # 统计
    rising = (quotes['change_pct'] > 0).sum()
    falling = (quotes['change_pct'] < 0).sum()
    avg_change = quotes['change_pct'].mean()
    std_change = quotes['change_pct'].std()

    print(f"\n📊 市场概况:")
    print(f"   上涨: {rising} 只 ({rising/len(quotes)*100:.1f}%)")
    print(f"   下跌: {falling} 只 ({falling/len(quotes)*100:.1f}%)")
    print(f"   平均涨跌: {avg_change:+.2f}%")
    print(f"   标准差: {std_change:.2f}%")
    print(f"   最佳: {quotes.iloc[0]['name']} ({quotes.iloc[0]['code']}) {quotes.iloc[0]['change_pct']:+.2f}%")
    print(f"   最差: {quotes.iloc[-1]['name']} ({quotes.iloc[-1]['code']}) {quotes.iloc[-1]['change_pct']:+.2f}%")

    # 保存
    quotes.to_csv(OUTPUT_DIR / 'today_quote.csv', index=False, encoding='utf-8-sig')
    print(f"\n📁 已保存: {OUTPUT_DIR / 'today_quote.csv'}")


if __name__ == '__main__':
    main()
