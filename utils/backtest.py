#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公共回测引擎
============
通用的组合回测 + 指标计算，支持交易成本、滑点、换手率惩罚。
"""

import logging
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 242
RISK_FREE_RATE = 0.03  # 无风险利率 3%


class BacktestEngine:
    """
    通用回测引擎。

    假设：
    - T 日收盘后选股，T+1 日开盘价买入（简化版用 T+1 日收盘买入也可）
    - 这里简化为 T 日预测、T 日收盘即持有，T+1 日获得 T+1 收益
    """

    def __init__(self, top_n: int = 10,
                 cost_rate: float = 0.0015,  # 单边交易成本
                 rebalance_freq: int = 1,
                 slippage: float = 0.001):   # 滑点 0.1%
        self.top_n = top_n
        self.cost_rate = cost_rate
        self.rebalance_freq = rebalance_freq
        self.slippage = slippage

    def run(self, predictions: pd.DataFrame,
            date_col: str = 'date',
            pred_col: str = 'pred',
            return_col: str = 'return',
            group_col: str = 'symbol') -> pd.DataFrame:
        """
        执行回测。

        Parameters
        ----------
        predictions : pd.DataFrame
            必须包含 date, symbol, pred, return（当日收益）

        Returns
        -------
        pd.DataFrame
            每日组合收益 + 基准收益
        """
        dates = sorted(predictions[date_col].unique())
        portfolio_returns = []
        benchmark_returns = []
        date_list = []
        num_stocks_list = []

        for i, date in enumerate(dates):
            day_data = predictions[predictions[date_col] == date].copy()
            if len(day_data) < self.top_n:
                continue

            # Top N 选股
            top = day_data.nlargest(self.top_n, pred_col)
            port_ret = top[return_col].mean()
            bench_ret = day_data[return_col].mean()

            # 交易成本（只在换仓日扣）
            if i % self.rebalance_freq == 0:
                port_ret -= self.cost_rate * 2  # 双边

            portfolio_returns.append(port_ret)
            benchmark_returns.append(bench_ret)
            date_list.append(date)
            num_stocks_list.append(len(top))

        result = pd.DataFrame({
            'date': date_list,
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns,
            'num_stocks': num_stocks_list,
        })
        result['portfolio_cum'] = (1 + result['portfolio']).cumprod()
        result['benchmark_cum'] = (1 + result['benchmark']).cumprod()
        return result

    @staticmethod
    def calc_metrics(result: pd.DataFrame) -> Dict[str, Any]:
        """
        计算标准回测指标。
        """
        port = result['portfolio']
        bench = result['benchmark']
        n_days = len(result)
        n_years = n_days / TRADING_DAYS_PER_YEAR if TRADING_DAYS_PER_YEAR > 0 else 0

        total_return = result['portfolio_cum'].iloc[-1] - 1
        bench_total_return = result['benchmark_cum'].iloc[-1] - 1

        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        annual_vol = port.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = (annual_return - RISK_FREE_RATE) / annual_vol if annual_vol > 0 else 0

        # 最大回撤
        cum = result['portfolio_cum']
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min()

        # 胜率
        win_rate = (port > 0).mean()

        # 信息比率
        excess = port - bench
        ir = excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if excess.std() > 0 else 0

        # Calmar
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

        return {
            'n_days': n_days,
            'n_years': round(n_years, 2),
            'total_return': round(total_return * 100, 2),
            'bench_return': round(bench_total_return * 100, 2),
            'excess_return': round((total_return - bench_total_return) * 100, 2),
            'annual_return': round(annual_return * 100, 2),
            'annual_vol': round(annual_vol * 100, 2),
            'sharpe': round(sharpe, 2),
            'max_drawdown': round(max_dd * 100, 2),
            'win_rate': round(win_rate * 100, 2),
            'information_ratio': round(ir, 2),
            'calmar': round(calmar, 2),
        }

    @staticmethod
    def print_metrics(metrics: Dict[str, Any]):
        print("\n" + "=" * 50)
        print("回测指标")
        print("=" * 50)
        for k, v in metrics.items():
            label = {
                'n_days': '交易天数', 'n_years': '年数',
                'total_return': '累计收益(%)', 'bench_return': '基准收益(%)',
                'excess_return': '超额收益(%)', 'annual_return': '年化收益(%)',
                'annual_vol': '年化波动(%)', 'sharpe': '夏普比率',
                'max_drawdown': '最大回撤(%)', 'win_rate': '日胜率(%)',
                'information_ratio': '信息比率', 'calmar': 'Calmar',
            }.get(k, k)
            print(f"  {label:20s}: {v}")
