#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多因子 + LightGBM 策略（腾讯1年真实数据版）
==========================================
用于快速验证，数据量较小，训练窗口也缩短。
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
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.api import TencentHistoryAPI
from utils.factors import FACTOR_COLS, calc_factors, build_factor_dataset
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

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
}


def fetch_1year_data():
    """获取最近1年数据。"""
    end = datetime.now()
    start = end - timedelta(days=365)
    start_str, end_str = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')

    logger.info(f"获取1年数据: {start_str} ~ {end_str}")
    results = TencentHistoryAPI.get_multi_kline(
        list(STOCK_POOL.keys()), start_str, end_str,
        period='day', adjust='qfq', delay=0.3
    )
    if not results:
        raise ValueError("无有效数据")
    return results


def train_and_backtest(kline_dict: dict):
    """训练 + 回测（前7个月训练，后5个月测试）。"""
    factor_df = build_factor_dataset(kline_dict, min_bars=60, target_horizon=3)

    all_dates = sorted(factor_df['date'].unique())
    split_idx = int(len(all_dates) * 0.6)
    train_end = all_dates[split_idx]

    train_df = factor_df[factor_df['date'] <= train_end]
    test_df = factor_df[factor_df['date'] > train_end]

    # 60% 训练 / 20% 验证 / 20% 测试
    split_valid = int(len(all_dates) * 0.80)
    valid_end = all_dates[split_valid]
    valid_df = factor_df[(factor_df['date'] > train_end) & (factor_df['date'] <= valid_end)]
    test_df = factor_df[factor_df['date'] > valid_end]

    X_train, y_train = train_df[FACTOR_COLS], train_df['target']
    X_valid, y_valid = valid_df[FACTOR_COLS], valid_df['target']
    logger.info(f"训练: {len(train_df)} 条 ({train_df['date'].min().date()} ~ {train_df['date'].max().date()})")
    logger.info(f"验证: {len(valid_df)} 条 ({valid_df['date'].min().date()} ~ {valid_df['date'].max().date()})")
    logger.info(f"测试:  {len(test_df)} 条 ({test_df['date'].min().date()} ~ {test_df['date'].max().date()})")

    train_set = lgb.Dataset(X_train, y_train)
    valid_set = lgb.Dataset(X_valid, y_valid, reference=train_set)
    model = lgb.train(LGB_PARAMS, train_set,
                      num_boost_round=200,
                      valid_sets=[train_set, valid_set],
                      valid_names=['train', 'valid'],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(period=50)])

    fi = pd.DataFrame({
        'factor': FACTOR_COLS,
        'importance': model.feature_importance(importance_type='gain'),
    }).sort_values('importance', ascending=False)

    X_test = test_df[FACTOR_COLS]
    test_df = test_df.copy()
    test_df['pred'] = model.predict(X_test)

    engine = BacktestEngine(top_n=10, cost_rate=0.0015)
    result = engine.run(test_df, date_col='date', pred_col='pred', return_col='next_return', group_col='symbol')
    metrics = engine.calc_metrics(result)
    return result, metrics, fi


def plot(result: pd.DataFrame, metrics: dict, fi: pd.DataFrame, filename: Path):
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    ax1.plot(result['date'], result['portfolio_cum'], label='组合', color='#E74C3C', linewidth=1.5)
    ax1.plot(result['date'], result['benchmark_cum'], label='基准', color='#3498DB', linewidth=1.2, linestyle='--')
    ax1.set_title('累计收益', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    cum = result['portfolio_cum']
    dd = (cum - cum.cummax()) / cum.cummax() * 100
    ax2.fill_between(result['date'], dd, 0, color='#E74C3C', alpha=0.4)
    ax2.set_title(f'回撤 (最大{metrics["max_drawdown"]:.1f}%)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.hist(result['portfolio'] * 100, bins=30, alpha=0.7, color='#3498DB', edgecolor='white')
    ax3.axvline(x=0, color='red', linestyle='--')
    ax3.set_title('日收益分布', fontsize=12, fontweight='bold')

    ax4 = axes[1, 1]
    top10 = fi.head(10)
    ax4.barh(range(len(top10)), top10['importance'], color='#2ECC71', alpha=0.8)
    ax4.set_yticks(range(len(top10)))
    ax4.set_yticklabels(top10['factor'])
    ax4.invert_yaxis()
    ax4.set_title('特征重要性', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"图表: {filename}")


def main():
    print("\n" + "=" * 60)
    print("  多因子 + LightGBM（腾讯1年真实数据）")
    print("=" * 60 + "\n")

    kline_dict = fetch_1year_data()
    result, metrics, fi = train_and_backtest(kline_dict)

    print("\n" + "=" * 50)
    print("回测结果")
    print("=" * 50)
    BacktestEngine.print_metrics(metrics)

    chart_file = OUTPUT_DIR / 'tencent_1year_backtest.png'
    plot(result, metrics, fi, chart_file)

    output = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': metrics,
        'feature_importance': fi.to_dict('records'),
    }
    with open(OUTPUT_DIR / 'tencent_1year_result.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 完成: {OUTPUT_DIR / 'tencent_1year_result.json'}")


if __name__ == '__main__':
    main()
