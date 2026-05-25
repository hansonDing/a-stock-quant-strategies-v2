#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本面数据获取与合并
===================
从 akshare 获取业绩报表（PE、ROE、营收增长等），合并到因子数据集。
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# 基本面因子列
FUNDAMENTAL_COLS = ['pe', 'roe', 'revenue_growth']


def fetch_fundamentals(date: Optional[str] = None) -> pd.DataFrame:
    """
    获取全市场业绩报表数据。

    Parameters
    ----------
    date : str, optional
        财报期，如 '20241231'。None 则自动选最新。

    Returns
    -------
    pd.DataFrame
        columns: code, name, pe(来自comment), roe, revenue_growth, profit_growth, gross_margin
    """
    try:
        import akshare as ak
    except ImportError:
        logger.warning("缺少 akshare，基本面数据不可用")
        return pd.DataFrame(columns=['code', 'name', 'roe', 'revenue_growth', 'profit_growth', 'gross_margin'])

    # 尝试最新季度
    if date is None:
        # 优先 2025Q1， fallback 2024年报
        for q in ['20250331', '20241231', '20240930', '20240630']:
            try:
                df = ak.stock_yjbb_em(date=q)
                if len(df) > 1000:
                    logger.info(f"使用业绩报表: {q}, 共 {len(df)} 条")
                    break
            except Exception as e:
                logger.warning(f"{q} 获取失败: {e}")
                continue
        else:
            return pd.DataFrame(columns=['code', 'name', 'roe', 'revenue_growth', 'profit_growth', 'gross_margin'])
    else:
        try:
            df = ak.stock_yjbb_em(date=date)
        except Exception as e:
            logger.warning(f"业绩报表获取失败: {e}")
            return pd.DataFrame(columns=['code', 'name', 'roe', 'revenue_growth', 'profit_growth', 'gross_margin'])

    # 标准化列名
    rename_map = {
        '股票代码': 'code',
        '股票简称': 'name',
        '净资产收益率': 'roe',
        '营业总收入-同比增长': 'revenue_growth',
        '净利润-同比增长': 'profit_growth',
        '销售毛利率': 'gross_margin',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 清洗数值
    for col in ['roe', 'revenue_growth', 'profit_growth', 'gross_margin']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    cols = ['code', 'name'] + [c for c in ['roe', 'revenue_growth', 'profit_growth', 'gross_margin'] if c in df.columns]
    return df[cols].copy()


def merge_fundamentals(factor_df: pd.DataFrame, fund_df: pd.DataFrame,
                       symbol_col: str = 'symbol') -> pd.DataFrame:
    """
    将基本面数据合并到因子数据集。

    Parameters
    ----------
    factor_df : pd.DataFrame
        含 symbol 列的因子数据
    fund_df : pd.DataFrame
        fetch_fundamentals() 的输出
    symbol_col : str
        factor_df 中的股票代码列名

    Returns
    -------
    pd.DataFrame
        合并后的数据，基本面列缺失则填 NaN
    """
    if fund_df.empty or symbol_col not in factor_df.columns:
        # 没有基本面数据，添加空列
        for col in FUNDAMENTAL_COLS:
            factor_df[col] = np.nan
        return factor_df

    # 统一代码格式：腾讯接口用 sh600519/sz000001/bj920000，akshare 用 600519/000001
    # 两边都转成纯数字代码来匹配
    factor_df = factor_df.copy()
    factor_df['_merge_code'] = factor_df[symbol_col].astype(str).str.replace(r'^(sh|sz|bj)', '', regex=True)
    fund_df = fund_df.copy()
    fund_df['_merge_code'] = fund_df['code'].astype(str)

    # 合并
    merged = factor_df.merge(
        fund_df[['_merge_code'] + [c for c in FUNDAMENTAL_COLS if c in fund_df.columns]],
        on='_merge_code',
        how='left'
    )
    merged = merged.drop(columns=['_merge_code'])

    # 确保所有基本面列都存在
    for col in FUNDAMENTAL_COLS:
        if col not in merged.columns:
            merged[col] = np.nan

    # 记录合并成功率
    non_na = merged['roe'].notna().sum()
    total = len(merged)
    logger.info(f"基本面合并: {non_na}/{total} 条有数据 ({non_na/total*100:.1f}%)")

    return merged


def apply_fundamental_filter(df: pd.DataFrame,
                              pe_max: float = 100.0,
                              roe_min: float = 0.0,
                              revenue_growth_min: float = -50.0) -> pd.DataFrame:
    """
    基本面过滤。

    Returns
    -------
    pd.DataFrame
        过滤后的数据
    """
    before = len(df)
    mask = pd.Series(True, index=df.index)

    if 'pe' in df.columns and df['pe'].notna().any():
        # PE > 0 且 < pe_max（负PE = 亏损，也过滤）
        pe_mask = (df['pe'].isna()) | ((df['pe'] > 0) & (df['pe'] < pe_max))
        dropped_pe = (~pe_mask).sum()
        if dropped_pe > 0:
            logger.info(f"基本面过滤-PE: 剔除 {dropped_pe} 只 (PE<=0 或 PE>={pe_max})")
        mask &= pe_mask

    if 'roe' in df.columns and df['roe'].notna().any():
        roe_mask = (df['roe'].isna()) | (df['roe'] >= roe_min)
        dropped_roe = (~roe_mask).sum()
        if dropped_roe > 0:
            logger.info(f"基本面过滤-ROE: 剔除 {dropped_roe} 只 (ROE<{roe_min}%)")
        mask &= roe_mask

    if 'revenue_growth' in df.columns and df['revenue_growth'].notna().any():
        rev_mask = (df['revenue_growth'].isna()) | (df['revenue_growth'] >= revenue_growth_min)
        dropped_rev = (~rev_mask).sum()
        if dropped_rev > 0:
            logger.info(f"基本面过滤-营收: 剔除 {dropped_rev} 只 (营收增长<{revenue_growth_min}%)")
        mask &= rev_mask

    filtered = df[mask].copy()
    logger.info(f"基本面过滤: {before} → {len(filtered)} 只")
    return filtered
