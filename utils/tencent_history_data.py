#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
腾讯财经历史K线数据获取（兼容层）
==================================
已迁移到 utils/api.py 中的 TencentHistoryAPI 类。
本文件保留为兼容层，直接调用公共模块。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.api import TencentHistoryAPI  # noqa

if __name__ == '__main__':
    from datetime import datetime, timedelta
    import numpy as np
    import pandas as pd

    test_stocks = ['sh600519', 'sz000858', 'sh601318']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    print(f"时间范围: {start_date} ~ {end_date}")
    data_dict = TencentHistoryAPI.get_multi_kline(
        test_stocks, start_date, end_date, period='day', adjust='qfq', delay=0.3
    )

    for symbol, df in data_dict.items():
        print(f"\n【{symbol}】")
        print(df.tail(5)[['date', 'open', 'close', 'high', 'low', 'volume']].to_string(index=False))
        df['return'] = df['close'].pct_change()
        print(f"  累计收益: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:.2f}%")
        print(f"  日均收益: {df['return'].mean()*100:.3f}%")
        print(f"  波动率: {df['return'].std()*np.sqrt(252)*100:.2f}%")

    if data_dict:
        all_data = pd.concat(data_dict.values(), ignore_index=True)
        output_path = PROJECT_ROOT / 'data' / 'tencent_history_data.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        all_data.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n数据已保存: {output_path} ({len(all_data)} 条)")
