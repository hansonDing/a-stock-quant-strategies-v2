#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多因子 + LightGBM 策略回测（腾讯3年真实数据版）
=============================================
1. 拉取3年真实K线
2. 计算18个多因子
3. 滚动训练（walk-forward）
4. 完整回测（含交易成本）
5. 输出图表 + 指标
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


def fetch_data(years: int = 3, min_bars: int = 200):
    """获取所有股票多年K线。"""
    logger.info(f"获取 {years} 年数据，共 {len(STOCK_POOL)} 只股票...")
    all_data = []
    for i, (code, name) in enumerate(STOCK_POOL.items()):
        logger.info(f"[{i+1}/{len(STOCK_POOL)}] {name}({code}) ...")
        df = TencentHistoryAPI.get_nyear_data(code, years=years, delay=0.2)
        if df is not None and len(df) >= min_bars:
            all_data.append(df)
            logger.info(f"  ✓ {len(df)} 条 ({df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()})")
        else:
            logger.warning(f"  ✗ 失败 ({len(df) if df is not None else 0} 条)")
    if not all_data:
        raise ValueError("无有效数据")
    return pd.concat(all_data, ignore_index=True)


def rolling_walkforward_backtest(data: pd.DataFrame,
                                  train_window_days: int = 400,
                                  test_window_days: int = 60,
                                  top_n: int = 10,
                                  target_horizon: int = 5):
    """
    Walk-forward 滚动回测：每过一个窗口就重训一次模型。

    返回 predictions DataFrame 用于通用回测引擎。
    """
    dates = sorted(data['date'].unique())
    logger.info(f"总交易日: {len(dates)} ({dates[0].date()} ~ {dates[-1].date()})")

    all_predictions = []
    all_feature_importance = []

    start_idx = 0
    step = 0
    while start_idx + train_window_days + test_window_days < len(dates):
        train_end_idx = start_idx + train_window_days
        test_end_idx = train_end_idx + test_window_days

        train_dates = dates[start_idx:train_end_idx]
        test_dates = dates[train_end_idx:test_end_idx]

        # 截取这段时间的数据
        window_data = data[data['date'].isin(train_dates + test_dates)].copy()
        if window_data.empty:
            start_idx += test_window_days
            step += 1
            continue

        # 计算因子
        kline_dict = {sym: grp for sym, grp in window_data.groupby('symbol')}
        factor_df = build_factor_dataset(kline_dict, min_bars=80, target_horizon=target_horizon)

        train_df = factor_df[factor_df['date'].isin(train_dates)]
        test_df = factor_df[factor_df['date'].isin(test_dates)]

        if len(train_df) < 500 or len(test_df) < 100:
            start_idx += test_window_days
            step += 1
            continue

        # 训练
        X_train = train_df[FACTOR_COLS]
        y_train = train_df['target']
        model = lgb.train(LGB_PARAMS, lgb.Dataset(X_train, y_train), num_boost_round=200,
                          valid_sets=[lgb.Dataset(X_train, y_train)],
                          callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])

        # 记录特征重要性
        fi = pd.DataFrame({
            'factor': FACTOR_COLS,
            'importance': model.feature_importance(importance_type='gain'),
            'window': step,
        })
        all_feature_importance.append(fi)

        # 预测
        X_test = test_df[FACTOR_COLS]
        test_df = test_df.copy()
        test_df['pred'] = model.predict(X_test)
        all_predictions.append(test_df)

        logger.info(f"  窗口 {step+1}: 训练 {len(train_df)} 条, 测试 {len(test_df)} 条, "
                    f"日期 {test_dates[0].date()} ~ {test_dates[-1].date()}")

        start_idx += test_window_days
        step += 1

    if not all_predictions:
        raise ValueError("滚动回测无有效预测")

    predictions = pd.concat(all_predictions, ignore_index=True)
    fi_all = pd.concat(all_feature_importance, ignore_index=True)
    fi_mean = fi_all.groupby('factor')['importance'].mean().reset_index().sort_values('importance', ascending=False)

    return predictions, fi_mean


def plot_results(result: pd.DataFrame, metrics: dict, fi_mean: pd.DataFrame, filename: Path):
    """绘制回测结果图表。"""
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 累计收益
    ax1 = axes[0, 0]
    ax1.plot(result['date'], result['portfolio_cum'], label='多因子组合', color='#E74C3C', linewidth=1.5)
    ax1.plot(result['date'], result['benchmark_cum'], label='基准(等权)', color='#3498DB', linewidth=1.2, linestyle='--')
    ax1.set_title('累计收益对比', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 回撤
    ax2 = axes[0, 1]
    cum = result['portfolio_cum']
    dd = (cum - cum.cummax()) / cum.cummax() * 100
    ax2.fill_between(result['date'], dd, 0, color='#E74C3C', alpha=0.4)
    ax2.set_title(f'回撤 (最大{metrics["max_drawdown"]:.1f}%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('回撤 (%)')
    ax2.grid(True, alpha=0.3)

    # 日收益分布
    ax3 = axes[1, 0]
    ax3.hist(result['portfolio'] * 100, bins=30, alpha=0.7, color='#3498DB', edgecolor='white')
    ax3.axvline(x=0, color='red', linestyle='--')
    ax3.set_title('日收益分布', fontsize=12, fontweight='bold')
    ax3.set_xlabel('日收益 (%)')

    # 特征重要性
    ax4 = axes[1, 1]
    top10 = fi_mean.head(10)
    ax4.barh(range(len(top10)), top10['importance'], color='#2ECC71', alpha=0.8)
    ax4.set_yticks(range(len(top10)))
    ax4.set_yticklabels(top10['factor'])
    ax4.invert_yaxis()
    ax4.set_title('特征重要性 Top 10', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"图表已保存: {filename}")


def main():
    print("\n" + "=" * 70)
    print("  多因子 + LightGBM 策略（腾讯3年真实数据 + Walk-Forward 回测）")
    print("=" * 70 + "\n")

    # 1. 获取数据
    data = fetch_data(years=3, min_bars=200)

    # 2. Walk-forward 滚动回测
    print("\n" + "=" * 70)
    print("Walk-Forward 滚动回测...")
    print("=" * 70)
    predictions, fi_mean = rolling_walkforward_backtest(
        data,
        train_window_days=400,
        test_window_days=60,
        top_n=10,
        target_horizon=5,
    )

    # 3. 通用回测引擎
    engine = BacktestEngine(top_n=10, cost_rate=0.0015)
    result = engine.run(predictions, date_col='date', pred_col='pred', return_col='next_return', group_col='symbol')
    metrics = engine.calc_metrics(result)

    print("\n" + "=" * 50)
    print("回测结果")
    print("=" * 50)
    BacktestEngine.print_metrics(metrics)

    # 4. 绘图
    chart_file = OUTPUT_DIR / 'tencent_3year_backtest.png'
    plot_results(result, metrics, fi_mean, chart_file)

    # 5. 保存
    output = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': metrics,
        'feature_importance': fi_mean.to_dict('records'),
    }
    with open(OUTPUT_DIR / 'tencent_3year_result.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 结果保存至: {OUTPUT_DIR / 'tencent_3year_result.json'}")


if __name__ == '__main__':
    main()
