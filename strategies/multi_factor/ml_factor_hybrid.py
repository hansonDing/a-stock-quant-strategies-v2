#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚠️ 混合策略演示（模拟训练 + 实时预测，仅供学习）
==================================================
本文件演示了"用模拟数据训练 + 实时行情预测"的流程，
但模型未接触真实A股历史数据，预测结果无实盘参考价值。

如需真实预测，请运行:
  - daily_stock_picks.py
  - strategies/multi_factor/tomorrow_picks.py
"""

import os
import sys
import logging
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.api import TencentQuoteAPI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / 'data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

STOCK_POOL = {
    'sh600519': '贵州茅台', 'sz000858': '五粮液', 'sh601318': '中国平安',
    'sz000333': '美的集团', 'sh600036': '招商银行', 'sh600276': '恒瑞医药',
    'sh601888': '中国中免', 'sz002594': '比亚迪', 'sh603259': '药明康德',
    'sz300750': '宁德时代', 'sh600030': '中信证券', 'sz000001': '平安银行',
    'sh601012': '隆基绿能', 'sz002415': '海康威视', 'sh600887': '伊利股份',
    'sh601166': '兴业银行', 'sz000568': '泸州老窖', 'sh600309': '万华化学',
    'sz002142': '宁波银行', 'sh603288': '海天味业', 'sz300059': '东方财富',
    'sh601668': '中国建筑', 'sz000002': '万科A', 'sh600900': '长江电力',
    'sz300760': '迈瑞医疗', 'sh601398': '工商银行', 'sz002230': '科大讯飞',
    'sh600028': '中国石化', 'sz000063': '中兴通讯',
}

FACTOR_COLS = [
    'mom_5', 'mom_10', 'mom_20', 'mom_accel', 'vol_5', 'vol_20', 'vol_ratio',
    'bias_5', 'bias_20', 'rsi_14', 'macd_dif', 'macd_dea', 'macd_hist',
    'turnover_ma5', 'amihud',
]


def generate_training_data(n_stocks=50, n_days=500, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='B')
    stock_names = [f"股票{i+1:03d}" for i in range(n_stocks)]
    base_prices = np.random.uniform(10, 200, n_stocks)
    betas = np.random.uniform(0.5, 1.5, n_stocks)
    quality_scores = np.random.uniform(0.05, 0.35, n_stocks)
    market_returns = np.random.normal(0.0001, 0.012, n_days)
    market_returns += np.sin(np.linspace(0, 6 * np.pi, n_days)) * 0.002

    all_data = []
    for i in range(n_stocks):
        noise = np.random.normal(0, 0.018, n_days)
        stock_returns = betas[i] * market_returns + noise + quality_scores[i] * 0.0003
        for t in range(20, n_days):
            stock_returns[t] += np.mean(stock_returns[t - 20:t]) * 0.05
        prices = base_prices[i] * np.cumprod(1 + stock_returns)
        volumes = np.random.randint(100000, 2000000, n_days)
        df = pd.DataFrame({
            'datetime': dates, 'stock': stock_names[i],
            'close': prices, 'volume': volumes,
        })
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)


def calc_factors(df):
    df = df.copy().sort_values('datetime').reset_index(drop=True)
    close, volume = df['close'], df['volume']
    df['return'] = close.pct_change()
    df['mom_5'] = close.pct_change(5)
    df['mom_10'] = close.pct_change(10)
    df['mom_20'] = close.pct_change(20)
    df['mom_accel'] = df['mom_5'] - df['mom_20']
    df['vol_5'] = df['return'].rolling(5).std()
    df['vol_20'] = df['return'].rolling(20).std()
    df['vol_ratio'] = df['vol_5'] / (df['vol_20'] + 1e-8)
    df['ma5'] = close.rolling(5).mean()
    df['ma20'] = close.rolling(20).mean()
    df['bias_5'] = close / df['ma5'] - 1
    df['bias_20'] = close / df['ma20'] - 1
    delta = close.diff()
    gain, loss = delta.clip(lower=0), (-delta).clip(lower=0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    ema12, ema26 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
    df['macd_dif'] = ema12 - ema26
    df['macd_dea'] = df['macd_dif'].ewm(span=9).mean()
    df['macd_hist'] = df['macd_dif'] - df['macd_dea']
    df['turnover'] = volume / (close * 10000 + 1)
    df['turnover_ma5'] = df['turnover'].rolling(5).mean()
    df['amihud'] = (df['return'].abs() / (volume + 1)).rolling(20).mean()
    df['target'] = df['return'].shift(-5).rolling(5).sum()
    return df


def main():
    print("⚠️  混合策略演示（模拟训练 + 实时预测）")
    print("    仅供学习，预测结果无实盘参考价值！\n")

    # 模拟训练
    data = generate_training_data(n_stocks=50, n_days=500, seed=42)
    all_features = []
    for stock, group in data.groupby('stock'):
        all_features.append(calc_factors(group))
    features_df = pd.concat(all_features, ignore_index=True)
    features_df = features_df.dropna(subset=FACTOR_COLS + ['target'])

    X, y = features_df[FACTOR_COLS], features_df['target']
    params = {
        'objective': 'regression', 'metric': 'mse', 'boosting_type': 'gbdt',
        'num_leaves': 31, 'learning_rate': 0.05, 'verbose': -1,
    }
    model = lgb.train(params, lgb.Dataset(X, y), num_boost_round=100)

    # 实时行情（真实）
    quotes = TencentQuoteAPI.get_quotes(list(STOCK_POOL.keys()))
    for col in ['close', 'open', 'high', 'low', 'volume', 'amount', 'change_pct', 'turnover', 'pe']:
        if col in quotes.columns:
            quotes[col] = pd.to_numeric(quotes[col], errors='coerce')

    # ⚠️ 用今日数据近似因子（仅供演示，实际应取历史K线计算）
    quotes['mom_5'] = quotes['change_pct'] / 100 * 0.5
    quotes['mom_10'] = quotes['change_pct'] / 100 * 0.3
    quotes['mom_20'] = quotes['change_pct'] / 100 * 0.2
    quotes['mom_accel'] = quotes['mom_5'] - quotes['mom_20']
    quotes['vol_5'] = quotes['turnover'] / 100
    quotes['vol_20'] = quotes['turnover'] / 100 * 0.8
    quotes['vol_ratio'] = 1.0
    quotes['bias_5'] = quotes['change_pct'] / 1000
    quotes['bias_20'] = quotes['change_pct'] / 1000 * 0.5
    quotes['rsi_14'] = (50 + quotes['change_pct'] * 0.5).clip(10, 90)
    quotes['macd_dif'] = quotes['change_pct'] / 100
    quotes['macd_dea'] = quotes['change_pct'] / 100 * 0.8
    quotes['macd_hist'] = quotes['macd_dif'] - quotes['macd_dea']
    quotes['turnover_ma5'] = quotes['turnover'] / 100
    quotes['amihud'] = (quotes['change_pct'].abs() / 100) / (quotes['volume'] + 1)

    X_pred = quotes[FACTOR_COLS].fillna(0)
    quotes['pred_return'] = model.predict(X_pred)
    quotes = quotes.sort_values('pred_return', ascending=False)

    print(f"\nTop 10 选股推荐（⚠️ 模拟模型，仅供参考）:")
    for i, (_, row) in enumerate(quotes.head(10).iterrows(), 1):
        print(f"{i:2d}. {row.get('name', row['code_key']):<10}  "
              f"预测: {row['pred_return']*100:+.2f}%  "
              f"现价: {row['close']:.2f}  "
              f"涨跌: {row['change_pct']:+.2f}%")


if __name__ == '__main__':
    main()
