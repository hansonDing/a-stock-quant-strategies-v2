#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公共因子计算模块
================
统一的多因子计算逻辑，避免在多个文件中重复实现。
"""

import logging
from typing import List

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# 标准因子列表（18个）
FACTOR_COLS = [
    'mom_5', 'mom_10', 'mom_20', 'mom_60', 'mom_accel',
    'vol_5', 'vol_10', 'vol_20', 'vol_ratio',
    'rsi_14', 'macd_dif', 'macd_dea', 'macd_hist',
    'bias_5', 'bias_10', 'bias_20', 'bias_60',
    'turnover_ma5', 'amihud',
    'pe', 'roe', 'revenue_growth',
]


def calc_factors(df: pd.DataFrame, need_target: bool = True, target_horizon: int = 5) -> pd.DataFrame:
    """
    计算完整多因子。

    Parameters
    ----------
    df : pd.DataFrame
        单只股票的日线数据，必须包含 date, open, close, low, high, volume, amount
    need_target : bool
        是否计算目标变量（未来收益）
    target_horizon : int
        预测窗口天数（默认 5 日）

    Returns
    -------
    pd.DataFrame
        添加因子列后的 DataFrame
    """
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)

    required = {'date', 'close', 'volume'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少必需列: {missing}")

    close = df['close']
    volume = df['volume']
    df['return'] = close.pct_change()
    ret = df['return']

    # ---------- 动量因子 ----------
    df['mom_5'] = close.pct_change(5)
    df['mom_10'] = close.pct_change(10)
    df['mom_20'] = close.pct_change(20)
    df['mom_60'] = close.pct_change(60)
    df['mom_accel'] = df['mom_5'] - df['mom_20']

    # ---------- 波动率因子 ----------
    df['vol_5'] = ret.rolling(5).std()
    df['vol_10'] = ret.rolling(10).std()
    df['vol_20'] = ret.rolling(20).std()
    df['vol_ratio'] = df['vol_5'] / (df['vol_10'] + 1e-8)

    # ---------- RSI ----------
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # ---------- MACD ----------
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd_dif'] = ema12 - ema26
    df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_dif'] - df['macd_dea']

    # ---------- 均线偏离 ----------
    df['ma5'] = close.rolling(5).mean()
    df['ma10'] = close.rolling(10).mean()
    df['ma20'] = close.rolling(20).mean()
    df['ma60'] = close.rolling(60).mean()
    df['bias_5'] = close / df['ma5'] - 1
    df['bias_10'] = close / df['ma10'] - 1
    df['bias_20'] = close / df['ma20'] - 1
    df['bias_60'] = close / df['ma60'] - 1

    # ---------- 流动性因子 ----------
    # turnover = volume / (close * 流通股本)，这里用简化版：成交量 / 收盘价 * 10000
    df['turnover_proxy'] = volume / (close * 10000 + 1)
    df['turnover_ma5'] = df['turnover_proxy'].rolling(5).mean()
    df['amihud'] = (ret.abs() / (volume + 1)).rolling(20).mean()

    # ---------- 当日收益 ----------
    df['return'] = ret
    # 次日收益（回测用，避免未来函数）
    df['next_return'] = ret.shift(-1)

    # ---------- 目标变量 ----------
    if need_target:
        # 未来 target_horizon 日的累计收益 (T+1 ~ T+horizon)
        df['target'] = ret.shift(-1).rolling(target_horizon).sum()

    return df


def calc_latest_factors(df: pd.DataFrame) -> pd.Series:
    """
    计算最新一天的因子值并返回 Series。
    用于明日选股等实时预测场景。
    """
    df = calc_factors(df, need_target=False)
    latest = df.iloc[-1]
    # 只保留因子列
    factor_values = {k: latest.get(k, 0) for k in FACTOR_COLS}
    return pd.Series(factor_values)


def build_factor_dataset(kline_dict: dict, min_bars: int = 80,
                         target_horizon: int = 5) -> pd.DataFrame:
    """
    从多只股票 K 线字典构建完整因子数据集。

    Parameters
    ----------
    kline_dict : dict
        {symbol: DataFrame}
    min_bars : int
        单只股票最少需要多少根 K 线才参与
    target_horizon : int
        预测窗口

    Returns
    -------
    pd.DataFrame
        包含所有因子和目标变量的面板数据
    """
    all_features = []
    for symbol, df in kline_dict.items():
        if df is None or len(df) < min_bars:
            logger.warning(f"[{symbol}] 数据不足 {min_bars} 条，跳过")
            continue
        try:
            factors = calc_factors(df, need_target=True, target_horizon=target_horizon)
            all_features.append(factors)
        except Exception as e:
            logger.error(f"[{symbol}] 因子计算失败: {e}")
            continue

    if not all_features:
        raise ValueError("没有一只股票能成功计算因子")

    combined = pd.concat(all_features, ignore_index=True)
    # 剔除因子或目标含 NaN 的行
    needed_cols = FACTOR_COLS + ['target']
    before = len(combined)
    combined = combined.dropna(subset=[c for c in needed_cols if c in combined.columns])
    after = len(combined)
    logger.info(f"因子数据集: {after} 条有效样本 (剔除 {before - after} 条 NaN)")
    return combined
