#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚠️ 模拟数据版回测（仅供学习演示，不用于实盘）
=============================================
本文件使用随机生成的模拟K线数据进行回测，结果不代表真实A股表现。
如需真实数据回测，请运行:
  - strategies/multi_factor/ml_factor_tencent_1year.py
  - strategies/multi_factor/ml_factor_tencent_3year.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FACTOR_COLS = [
    'mom_5', 'mom_10', 'mom_20', 'mom_60', 'mom_accel',
    'vol_5', 'vol_20', 'vol_change', 'vol_ratio',
    'rsi_14', 'macd_dif', 'macd_dea', 'macd_hist',
    'bias_5', 'bias_20', 'bias_60',
    'turnover', 'turnover_ma5', 'turnover_change', 'amihud',
    'ep_ratio', 'bp_ratio', 'roe', 'profit_stability',
]


def generate_multi_stock_data(n_stocks=50, start_date="2022-01-04", end_date=None, seed=42):
    """生成模拟的多标的时间序列数据"""
    np.random.seed(seed)
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)

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
            stock_returns[t] += np.mean(stock_returns[t-20:t]) * 0.05
        for t in range(60, n_days):
            stock_returns[t] -= np.mean(stock_returns[t-60:t-20]) * 0.03

        prices = base_prices[i] * np.cumprod(1 + stock_returns)
        volumes = np.random.randint(100000, 2000000, n_days)

        df = pd.DataFrame({
            '日期': dates, '股票': stock_names[i], 'close': prices,
            'volume': volumes.astype(int), 'return': stock_returns,
            'beta': betas[i], 'quality': quality_scores[i],
        })
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


def build_factors(data):
    """构建多因子特征"""
    df = data.copy()
    df = df.sort_values(['股票', '日期']).reset_index(drop=True)
    factors = []

    for stock, group in df.groupby('股票'):
        g = group.copy().sort_values('日期').reset_index(drop=True)
        close, ret, vol = g['close'], g['return'], g['volume']

        g['mom_5'] = close.pct_change(5)
        g['mom_10'] = close.pct_change(10)
        g['mom_20'] = close.pct_change(20)
        g['mom_60'] = close.pct_change(60)
        g['mom_accel'] = g['mom_5'] - g['mom_20']
        g['vol_5'] = ret.rolling(5).std()
        g['vol_20'] = ret.rolling(20).std()
        g['vol_change'] = g['vol_5'] - g['vol_20']
        g['vol_ratio'] = g['vol_5'] / (g['vol_20'] + 1e-8)

        delta = close.diff()
        gain, loss = delta.clip(lower=0), (-delta).clip(lower=0)
        rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-8)
        g['rsi_14'] = 100 - (100 / (1 + rs))

        ema12, ema26 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
        g['macd_dif'] = ema12 - ema26
        g['macd_dea'] = g['macd_dif'].ewm(span=9).mean()
        g['macd_hist'] = g['macd_dif'] - g['macd_dea']

        g['ma5'], g['ma20'], g['ma60'] = close.rolling(5).mean(), close.rolling(20).mean(), close.rolling(60).mean()
        g['bias_5'] = close / g['ma5'] - 1
        g['bias_20'] = close / g['ma20'] - 1
        g['bias_60'] = close / g['ma60'] - 1

        g['turnover'] = vol / (close * 10000 + 1)
        g['turnover_ma5'] = g['turnover'].rolling(5).mean()
        g['turnover_change'] = g['turnover'] - g['turnover_ma5']
        g['amihud'] = (ret.abs() / (vol + 1)).rolling(20).mean()
        g['ep_ratio'] = g['quality'] * np.random.uniform(0.8, 1.2, len(g))
        g['bp_ratio'] = np.random.uniform(0.3, 2.0, len(g))
        g['roe'] = g['quality'] + np.random.normal(0, 0.02, len(g))
        g['profit_stability'] = ret.rolling(20).std().apply(lambda x: 1/(x+0.01))
        g['target_5d'] = g['return'].shift(-5).rolling(5).sum()

        factors.append(g)

    return pd.concat(factors, ignore_index=True)


def main():
    print("=" * 60)
    print("  ⚠️  多因子选股 + LightGBM 回测系统（模拟数据版）")
    print("  仅供学习演示，不用于实盘！")
    print("=" * 60)

    data = generate_multi_stock_data(n_stocks=50, start_date="2022-01-04", end_date="2026-05-06", seed=42)
    data = build_factors(data)
    factor_cols = FACTOR_COLS

    df = data.dropna(subset=factor_cols + ['target_5d'])
    dates = np.sort(df['日期'].unique())
    split_idx = int(len(dates) * 0.6)
    train_dates, test_dates = dates[:split_idx], dates[split_idx:]

    train_data = df[df['日期'].isin(train_dates)]
    X_train, y_train = train_data[factor_cols], train_data['target_5d']

    params = {
        'objective': 'regression', 'metric': 'mse', 'boosting_type': 'gbdt',
        'num_leaves': 31, 'learning_rate': 0.05, 'verbose': -1,
    }
    model = lgb.train(params, lgb.Dataset(X_train, y_train), num_boost_round=200,
                      valid_sets=[lgb.Dataset(X_train, y_train)],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

    importance = pd.DataFrame({
        'factor': factor_cols,
        'importance': model.feature_importance(importance_type='gain'),
    }).sort_values('importance', ascending=False)

    print("\nTop 10 重要因子:")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['factor']:20s}: {row['importance']:.1f}")

    # 回测
    predictions = []
    for date in test_dates:
        day_data = df[df['日期'] == date].copy()
        if len(day_data) == 0: continue
        day_data['pred'] = model.predict(day_data[factor_cols])
        predictions.append(day_data)

    pred_df = pd.concat(predictions, ignore_index=True)

    portfolio_returns, benchmark_returns, dates_list = [], [], []
    for date in test_dates:
        day_data = pred_df[pred_df['日期'] == date]
        if len(day_data) < 10: continue
        top = day_data.nlargest(10, 'pred')
        portfolio_returns.append(top['return'].mean())
        benchmark_returns.append(day_data['return'].mean())
        dates_list.append(date)

    result = pd.DataFrame({'date': dates_list, 'portfolio': portfolio_returns, 'benchmark': benchmark_returns})
    result['portfolio_cum'] = (1 + result['portfolio']).cumprod()
    result['benchmark_cum'] = (1 + result['benchmark']).cumprod()

    total_return = result['portfolio_cum'].iloc[-1] - 1
    bench_return = result['benchmark_cum'].iloc[-1] - 1
    n_years = len(result) / 242
    annual_return = (1 + total_return) ** (1/n_years) - 1
    annual_vol = result['portfolio'].std() * np.sqrt(242)
    sharpe = (annual_return - 0.03) / annual_vol
    max_dd = ((result['portfolio_cum'] - result['portfolio_cum'].cummax()) / result['portfolio_cum'].cummax()).min()

    print(f"\n累计收益: {total_return*100:.2f}%  年化: {annual_return*100:.2f}%  夏普: {sharpe:.2f}  最大回撤: {max_dd*100:.2f}%")

    output = {
        'metrics': {
            'total_return': round(total_return*100, 2),
            'annual_return': round(annual_return*100, 2),
            'sharpe': round(sharpe, 2),
            'max_drawdown': round(max_dd*100, 2),
        },
        'feature_importance': importance.to_dict('records'),
    }
    json_file = os.path.join(OUTPUT_DIR, 'ml_factor_result.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {json_file}")


if __name__ == '__main__':
    main()
