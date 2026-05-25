#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
明日选股推荐（真实数据版）
=========================
基于 daily_stock_picks.py 的模型做明日预测，支持单独运行。
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.api import TencentQuoteAPI, TencentHistoryAPI
from utils.factors import FACTOR_COLS, calc_factors

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / 'data' / 'daily_picks'
MODEL_PATH = OUTPUT_DIR / 'lgb_model.pkl'

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


def get_model() -> lgb.Booster:
    """加载或提示训练。"""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    raise FileNotFoundError(
        f"模型不存在: {MODEL_PATH}\n"
        f"请先运行: python daily_stock_picks.py --train"
    )


def predict() -> pd.DataFrame:
    model = get_model()
    logger.info("获取实时行情...")
    quotes = TencentQuoteAPI.get_quotes(list(STOCK_POOL.keys()))
    if quotes.empty:
        raise RuntimeError("获取行情失败")
    quotes['name'] = quotes['code_key'].map(STOCK_POOL)

    logger.info("获取历史数据计算因子...")
    all_factors = []
    for code, name in STOCK_POOL.items():
        df = TencentHistoryAPI.get_kline(
            code,
            (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d'),
            max_retries=1,
        )
        if df is None or len(df) < 60:
            logger.warning(f"[{code}] 数据不足，跳过")
            continue
        try:
            factors = calc_factors(df, need_target=False)
            latest = factors.iloc[-1]
            today = quotes[quotes['code_key'] == code]
            if len(today) > 0:
                latest['code'] = code
                latest['name'] = name
                latest['close'] = today.iloc[0]['close']
                latest['change_pct'] = today.iloc[0]['change_pct']
                latest['turnover'] = today.iloc[0]['turnover']
                latest['pe'] = today.iloc[0]['pe']
                all_factors.append(latest)
        except Exception as e:
            logger.warning(f"[{code}] 因子失败: {e}")

    features_df = pd.DataFrame(all_factors)
    X_pred = features_df[FACTOR_COLS].fillna(0)
    features_df['pred_return'] = model.predict(X_pred)
    return features_df.sort_values('pred_return', ascending=False)


def main():
    print("=" * 60)
    print("  明日选股推荐（真实数据版）")
    print(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    features_df = predict()
    top10 = features_df.head(10)

    print(f"\n{'='*60}")
    print("【明日选股推荐】Top 10")
    print(f"{'='*60}")
    print(f"{'排名':<6}{'名称':<10}{'代码':<12}{'预测收益':<12}{'今日收盘':<10}{'今日涨跌':<10}{'换手率':<8}{'PE':<8}")
    print("-" * 60)
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"{i:<6}{str(row['name'])[:8]:<10}{row['code']:<12}"
              f"{row['pred_return']*100:>+10.2f}%"
              f"{row.get('close',0):>9.2f}"
              f"{row.get('change_pct',0):>+9.2f}%"
              f"{row.get('turnover',0):>7.2f}%"
              f"{row.get('pe',0):>7.1f}")

    # 分析首选
    best = top10.iloc[0]
    print(f"\n🏆 首选: {best['name']} ({best['code']})")
    print(f"   预测收益: {best['pred_return']*100:+.2f}%")
    print(f"   今日收盘: {best['close']:.2f} ({best['change_pct']:+.2f}%)")
    print(f"   20日动量: {best.get('mom_20',0)*100:+.2f}%")
    print(f"   5日偏离:  {best.get('bias_5',0)*100:+.2f}%")
    print(f"   20日波动: {best.get('vol_20',0)*100:.2f}%")

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'top10': top10[['code','name','close','change_pct','pred_return','pe']].to_dict('records'),
    }
    with open(OUTPUT_DIR / 'tomorrow_picks.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 已保存: {OUTPUT_DIR / 'tomorrow_picks.json'}")


if __name__ == '__main__':
    main()
