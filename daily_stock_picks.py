#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股每日选股推荐（真实数据版）
=========================
使用腾讯历史数据 + 基本面因子训练 LightGBM 模型，预测次日涨跌。
流程：拉取3年真实K线 → 计算因子 → 合并基本面 → 训练模型 → 获取实时行情 → 预测排序 → 基本面过滤
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.api import TencentQuoteAPI, TencentHistoryAPI
from utils.factors import FACTOR_COLS, calc_factors, build_factor_dataset
from utils.fundamentals import (
    fetch_fundamentals, merge_fundamentals,
    apply_fundamental_filter, FUNDAMENTAL_COLS
)
from utils.backtest import BacktestEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / 'data' / 'daily_picks'
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = OUTPUT_DIR / 'lgb_model.pkl'

# ============================================================
# 股票池（主板蓝筹 + 主板科技 + 主板电力，剔除科创/创业/ST）
# 只保留 sh600/601/603 和 sz000/001/002
# ============================================================
STOCK_POOL = {
    # 蓝筹核心
    'sh600519': '贵州茅台', 'sz000858': '五粮液', 'sh601318': '中国平安',
    'sz000333': '美的集团', 'sh600036': '招商银行', 'sh600276': '恒瑞医药',
    'sh601888': '中国中免', 'sz002594': '比亚迪', 'sh603259': '药明康德',
    'sh600030': '中信证券', 'sz000001': '平安银行',
    'sz002415': '海康威视', 'sh600887': '伊利股份', 'sh601166': '兴业银行',
    'sz000568': '泸州老窖', 'sh600309': '万华化学',
    'sh601668': '中国建筑', 'sh600900': '长江电力',
    'sh601398': '工商银行', 'sz002230': '科大讯飞', 'sh600028': '中国石化',
    'sz000063': '中兴通讯',
    # 主板半导体
    'sh603501': '韦尔股份', 'sh603986': '兆易创新',
    'sz002371': '北方华创', 'sz002049': '紫光国微',
    'sh600584': '长电科技', 'sh600703': '三安光电',
    'sh600460': '士兰微', 'sh603019': '中科曙光',
    'sh603160': '汇顶科技', 'sh600745': '闻泰科技',
    'sh603290': '斯达半导', 'sz002156': '通富微电',
    # 软件/AI/计算机
    'sh600536': '中国软件', 'sh600570': '恒生电子',
    'sz000977': '浪潮信息', 'sz002236': '大华股份',
    'sz002410': '广联达', 'sz002405': '四维图新',
    # 通信
    'sh600498': '烽火通信', 'sh600487': '亨通光电',
    'sh600522': '中天科技',
    # 新能源科技
    'sz002459': '晶澳科技', 'sz002129': 'TCL中环',
    'sh600089': '特变电工', 'sh601877': '正泰电器',
    'sh601615': '明阳智能', 'sz002487': '大金重工',
    # 主板电力
    'sh600795': '国电电力', 'sh600011': '华能国际', 'sh601991': '大唐发电',
    'sh600886': '国投电力', 'sh601985': '中国核电', 'sh600027': '华电国际',
    'sz000539': '粤电力A', 'sz000883': '湖北能源',
    # 其他主板科技
    'sh601012': '隆基绿能', 'sh603688': '石英股份',
    'sh600438': '通威股份', 'sz002475': '立讯精密', 'sz002241': '歌尔股份',
    'sz000538': '云南白药', 'sh600436': '片仔癀',
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

# 成本控制参数
COST_RATE = 0.002       # 单边交易成本（佣金0.025% + 印花税0.1% + 滑点0.05% = 0.175%，取0.2%留余量）
SLIPPAGE = 0.001        # 额外滑点
PE_MAX = 100.0          # PE上限过滤
ROE_MIN = 0.0           # ROE下限过滤
REVENUE_GROWTH_MIN = -50.0  # 营收增长下限


def fetch_training_data(years: int = 3, min_bars: int = 200):
    """获取所有股票的多年K线数据。"""
    logger.info(f"获取 {years} 年历史数据，股票池共 {len(STOCK_POOL)} 只...")
    all_data = []
    for i, (code, name) in enumerate(STOCK_POOL.items()):
        logger.info(f"[{i+1}/{len(STOCK_POOL)}] {name}({code}) ...")
        df = TencentHistoryAPI.get_nyear_data(code, years=years, delay=0.2)
        if df is not None and len(df) >= min_bars:
            all_data.append(df)
            logger.info(f"  ✓ {len(df)} 条")
        else:
            logger.warning(f"  ✗ 失败或数据不足 ({len(df) if df is not None else 0} 条)")

    if not all_data:
        raise ValueError("无有效训练数据")

    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"\n共获取 {len(combined)} 条K线，{combined['symbol'].nunique()} 只股票")
    return combined


def train_model(data: pd.DataFrame, target_horizon: int = 1) -> lgb.Booster:
    """用真实历史数据 + 基本面因子训练 LightGBM。"""
    # 1. 计算技术面因子
    kline_dict = {sym: grp for sym, grp in data.groupby('symbol')}
    factor_df = build_factor_dataset(kline_dict, min_bars=80, target_horizon=target_horizon)

    # 2. 获取基本面数据并合并
    logger.info("获取基本面数据...")
    fund_df = fetch_fundamentals()
    if not fund_df.empty:
        factor_df = merge_fundamentals(factor_df, fund_df, symbol_col='symbol')

    # 3. 时间序列切分：60% 训练 / 20% 验证 / 20% 测试
    train_end = factor_df['date'].quantile(0.60)
    valid_end = factor_df['date'].quantile(0.80)

    train_df = factor_df[factor_df['date'] <= train_end]
    valid_df = factor_df[(factor_df['date'] > train_end) & (factor_df['date'] <= valid_end)]
    test_df  = factor_df[factor_df['date'] > valid_end]

    X_train = train_df[FACTOR_COLS]
    y_train = train_df['target']
    X_valid = valid_df[FACTOR_COLS]
    y_valid = valid_df['target']

    logger.info(f"训练: {len(train_df)} 条, {train_df['date'].min().date()} ~ {train_df['date'].max().date()}")
    logger.info(f"验证: {len(valid_df)} 条, {valid_df['date'].min().date()} ~ {valid_df['date'].max().date()}")
    logger.info(f"测试:  {len(test_df)} 条")

    train_set = lgb.Dataset(X_train, y_train)
    valid_set = lgb.Dataset(X_valid, y_valid, reference=train_set)

    model = lgb.train(
        LGB_PARAMS,
        train_set,
        num_boost_round=200,
        valid_sets=[train_set, valid_set],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(period=50)]
    )

    # 测试集评估
    if len(test_df) > 0:
        from sklearn.metrics import mean_squared_error
        test_pred = model.predict(test_df[FACTOR_COLS])
        test_mse = mean_squared_error(test_df['target'], test_pred)
        logger.info(f"测试集 MSE: {test_mse:.6f}")

    # 特征重要性
    importance = pd.DataFrame({
        'factor': FACTOR_COLS,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    logger.info("\nTop 10 重要因子:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['factor']:20s}: {row['importance']:.1f}")

    return model


def predict_tomorrow(model: lgb.Booster) -> pd.DataFrame:
    """获取实时行情 + 历史因子 + 基本面 → 预测明日涨跌。"""
    # 1. 实时行情
    logger.info("获取实时行情...")
    quotes = TencentQuoteAPI.get_quotes(list(STOCK_POOL.keys()))
    if quotes.empty:
        raise RuntimeError("获取实时行情失败")

    quotes['name'] = quotes['code_key'].map(STOCK_POOL)

    # 2. 剔除 ST 股票
    st_mask = quotes['name'].str.contains('ST', case=False, na=False)
    if st_mask.any():
        dropped_st = quotes[st_mask]['name'].tolist()
        logger.warning(f"剔除 ST 股票 {len(dropped_st)} 只: {', '.join(dropped_st)}")
        quotes = quotes[~st_mask].copy()

    # 3. 剔除涨停/接近涨停（次日无法买入）
    limit_up_mask = quotes['change_pct'] >= 9.5
    if limit_up_mask.any():
        dropped = quotes[limit_up_mask]['name'].tolist()
        logger.warning(f"剔除涨停 {len(dropped)} 只: {', '.join(dropped)}")
        quotes = quotes[~limit_up_mask].copy()

    # 3. 获取基本面数据
    logger.info("获取基本面数据...")
    fund_df = fetch_fundamentals()

    # 4. 逐个计算因子
    logger.info("获取历史数据计算因子...")
    all_factors = []
    for i, (code, name) in enumerate(STOCK_POOL.items()):
        if code not in quotes['code_key'].values:
            continue  # 已被过滤

        df = TencentHistoryAPI.get_kline(
            code,
            (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d'),
            max_retries=1,
        )
        if df is None or len(df) < 60:
            logger.warning(f"[{code}] 历史数据不足，跳过")
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

    if not all_factors:
        raise RuntimeError("没有一只股票能成功计算因子")

    features_df = pd.DataFrame(all_factors)

    # 5. 合并基本面
    if not fund_df.empty:
        features_df = merge_fundamentals(features_df, fund_df, symbol_col='code')

    # 6. 基本面过滤
    features_df = apply_fundamental_filter(
        features_df,
        pe_max=PE_MAX,
        roe_min=ROE_MIN,
        revenue_growth_min=REVENUE_GROWTH_MIN
    )

    if features_df.empty:
        raise RuntimeError("基本面过滤后无剩余股票")

    # 7. 预测
    X_pred = features_df[FACTOR_COLS].fillna(0)
    features_df['pred_return'] = model.predict(X_pred)
    features_df = features_df.sort_values('pred_return', ascending=False)
    return features_df


def generate_report(top10: pd.DataFrame, quotes: pd.DataFrame, report_date: datetime) -> str:
    today_str = report_date.strftime('%Y-%m-%d')
    tomorrow_str = (report_date + timedelta(days=1)).strftime('%Y-%m-%d')

    rising = (quotes['change_pct'] > 0).sum()
    falling = (quotes['change_pct'] < 0).sum()
    avg_change = quotes['change_pct'].mean()

    lines = [
        f"# A股每日选股推荐（科技+基本面增强版）",
        f"",
        f"> 生成时间: {report_date.strftime('%Y-%m-%d %H:%M:%S')}  |  推荐日期: {tomorrow_str}",
        f"> 模型: LightGBM + 21因子（18技术面 + 3基本面）| 预测窗口: 1日",
        f"> 成本控制: 单边{COST_RATE*100:.2f}% + 滑点{SLIPPAGE*100:.1f}%",
        f">",
        f"> ⚠️ **免责声明**: 本推荐仅供学习研究，不构成投资建议。投资有风险，入市需谨慎。",
        f"",
        f"---",
        f"",
        f"## 今日市场概况",
        f"",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| 上涨股票 | {rising} 只 |",
        f"| 下跌股票 | {falling} 只 |",
        f"| 平均涨跌 | {avg_change:+.2f}% |",
        f"",
        f"---",
        f"",
        f"## 明日 Top 10 选股推荐",
        f"",
        f"| 排名 | 股票 | 代码 | 预测涨跌 | 今日收盘 | 今日涨跌 | 换手 | PE | ROE | 营收增长 |",
        f"|:---:|------|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for i, (_, row) in enumerate(top10.iterrows(), 1):
        name = str(row['name'])[:8]
        code = row['code']
        pred = row['pred_return'] * 100
        close = row.get('close', 0)
        change = row.get('change_pct', 0)
        turnover = row.get('turnover', 0)
        pe = row.get('pe', 0)
        roe = row.get('roe', '-')
        rev = row.get('revenue_growth', '-')
        pe_str = f"{pe:.1f}" if pd.notna(pe) else "-"
        roe_str = f"{roe:.1f}%" if pd.notna(roe) else "-"
        rev_str = f"{rev:+.1f}%" if pd.notna(rev) else "-"
        lines.append(
            f"| {i} | **{name}** | {code} | {pred:+.2f}% | {close:.2f} | {change:+.2f}% | {turnover:.2f}% | {pe_str} | {roe_str} | {rev_str} |"
        )

    # 首选分析
    best = top10.iloc[0]
    pe_val = best.get('pe')
    pe_str = f"{pe_val:.1f}" if pd.notna(pe_val) else "-"
    pe_label = '偏高' if pd.notna(pe_val) and pe_val > 50 else ('合理' if pd.notna(pe_val) and pe_val > 0 else '亏损')
    roe_val = best.get('roe')
    roe_str = f"{roe_val:.1f}" if pd.notna(roe_val) else "-"
    roe_label = '优秀' if pd.notna(roe_val) and roe_val > 15 else ('良好' if pd.notna(roe_val) and roe_val > 5 else '一般')
    rev_val = best.get('revenue_growth')
    rev_str = f"{rev_val:+.1f}" if pd.notna(rev_val) else "-"
    rev_label = '高增长' if pd.notna(rev_val) and rev_val > 30 else ('增长' if pd.notna(rev_val) and rev_val > 0 else '下滑')

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 首选分析: {best['name']}",
        f"",
        f"| 因子 | 数值 | 解读 |",
        f"|------|:---:|------|",
        f"| 预测涨跌 | {best['pred_return']*100:+.2f}% | 模型预测明日表现 |",
        f"| 20日动量 | {best.get('mom_20',0)*100:+.2f}% | {'上涨' if best.get('mom_20',0)>0 else '下跌'}趋势 |",
        f"| 5日偏离 | {best.get('bias_5',0)*100:+.2f}% | {'偏高注意回调' if best.get('bias_5',0)>0.02 else '正常范围'} |",
        f"| RSI(14) | {best.get('rsi_14',0):.1f} | {'超买' if best.get('rsi_14',0)>70 else '超卖' if best.get('rsi_14',0)<30 else '中性'} |",
        f"| PE | {pe_str} | {pe_label} |",
        f"| ROE | {roe_str}% | {roe_label} |",
        f"| 营收增长 | {rev_str}% | {rev_label} |",
        f"",
        f"---",
        f"",
        f"## 风险提示",
        f"",
        f"1. 模型基于历史数据训练，过往表现不代表未来收益",
        f"2. 建议分散投资，单只股票仓位不超过10%",
        f"3. 设置止损线（如-5%），控制下行风险",
        f"4. 基本面数据来自最近财报，存在滞后性",
    ])

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='A股每日选股推荐（科技+基本面增强版）')
    parser.add_argument('--train', action='store_true', help='强制重新训练模型')
    parser.add_argument('--years', type=int, default=3, help='训练数据年数')
    parser.add_argument('--horizon', type=int, default=1, help='预测窗口天数（默认1日）')
    args = parser.parse_args()

    report_date = datetime.now()
    today_str = report_date.strftime('%Y-%m-%d')

    print(f"{'='*60}")
    print(f"  A股每日选股推荐（科技+基本面增强版）")
    print(f"  股票池: {len(STOCK_POOL)} 只（蓝筹+科技+存储+电力）")
    print(f"  预测窗口: {args.horizon} 日")
    print(f"  生成时间: {report_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # 加载或训练模型
    if MODEL_PATH.exists() and not args.train:
        print("\n[1/3] 加载已有模型...")
        model = joblib.load(MODEL_PATH)
        print("  ✓ 模型加载完成")
    else:
        print(f"\n[1/3] 训练模型（使用 {args.years} 年真实历史数据）...")
        data = fetch_training_data(years=args.years, min_bars=200)
        model = train_model(data, target_horizon=args.horizon)
        joblib.dump(model, MODEL_PATH)
        print(f"  ✓ 模型已保存: {MODEL_PATH}")

    # 预测
    print("\n[2/3] 获取行情并计算因子...")
    features_df = predict_tomorrow(model)
    top10 = features_df.head(10)
    print(f"  ✓ 成功预测 {len(features_df)} 只股票（已过滤）")

    # 打印结果
    print(f"\n{'='*60}")
    print("  明日 Top 10 选股推荐")
    print(f"{'='*60}")
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        pe = row.get('pe', '-')
        pe_str = f"PE:{pe:.1f}" if pd.notna(pe) else "PE:-"
        roe = row.get('roe', '-')
        roe_str = f"ROE:{roe:.1f}%" if pd.notna(roe) else "ROE:-"
        print(f"  {emoji} {i}. {row['name']:<8}  预测:{row['pred_return']*100:+.2f}%  "
              f"现价:{row['close']:.2f}  涨跌:{row['change_pct']:+.2f}%  {pe_str}  {roe_str}")

    # 生成报告
    print("\n[3/3] 生成报告...")
    quotes = TencentQuoteAPI.get_quotes(list(STOCK_POOL.keys()))
    quotes['name'] = quotes['code_key'].map(STOCK_POOL)
    report = generate_report(top10, quotes, report_date)
    report_file = OUTPUT_DIR / f"{today_str}_stock_picks.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    # 保存 JSON
    result = {
        'date': today_str,
        'top10': top10[['code','name','close','change_pct','pred_return','pe','roe','revenue_growth']].to_dict('records'),
    }
    json_file = OUTPUT_DIR / f"{today_str}_stock_picks.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 报告: {report_file}")
    print(f"✅ 数据: {json_file}")


if __name__ == '__main__':
    main()
