#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公共数据接口封装（腾讯财经）
====================
实时行情 + 历史K线，统一异常处理、重试、数据校验。
"""

import time
import logging
from typing import List, Optional, Dict, Any

import requests
import pandas as pd

logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# 实时行情
# -----------------------------------------------------------

class TencentQuoteAPI:
    """腾讯实时行情API封装"""

    BASE_URL = "https://qt.gtimg.cn/q="
    FIELDS = {
        0: 'market', 1: 'name', 2: 'code', 3: 'close', 4: 'pre_close',
        5: 'today_open', 6: 'volume', 7: 'bid_volume', 8: 'ask_volume',
        9: 'buy1_price', 10: 'buy1_volume', 11: 'buy2_price', 12: 'buy2_volume',
        13: 'buy3_price', 14: 'buy3_volume', 15: 'buy4_price', 16: 'buy4_volume',
        17: 'buy5_price', 18: 'buy5_volume', 19: 'sell1_price', 20: 'sell1_volume',
        21: 'sell2_price', 22: 'sell2_volume', 23: 'sell3_price', 24: 'sell3_volume',
        25: 'sell4_price', 26: 'sell4_volume', 27: 'sell5_price', 28: 'sell5_volume',
        29: 'recent_deal', 30: 'datetime', 31: 'change', 32: 'change_pct',
        33: 'high', 34: 'low', 35: 'price_volume', 36: 'volume_hand',
        37: 'amount', 38: 'turnover', 39: 'pe', 40: 'amplitude',
        41: 'high_limit', 42: 'low_limit', 43: 'total_cap', 44: 'float_cap', 45: 'pb',
        46: 'high_52w', 47: 'low_52w', 49: 'volume_ratio', 50: 'avg_price',
    }

    NUMERIC_COLS = ['close', 'pre_close', 'today_open', 'high', 'low', 'volume',
                    'change', 'change_pct', 'turnover', 'pe', 'pb', 'amount',
                    'amplitude', 'total_cap', 'float_cap', 'high_limit', 'low_limit',
                    'high_52w', 'low_52w', 'volume_ratio', 'avg_price',
                    'buy1_price', 'buy2_price', 'buy3_price', 'buy4_price', 'buy5_price',
                    'sell1_price', 'sell2_price', 'sell3_price', 'sell4_price', 'sell5_price',
                    'buy1_volume', 'buy2_volume', 'buy3_volume', 'buy4_volume', 'buy5_volume',
                    'sell1_volume', 'sell2_volume', 'sell3_volume', 'sell4_volume', 'sell5_volume']

    @classmethod
    def get_quotes(cls, codes: List[str], max_retries: int = 2, batch_size: int = 60) -> pd.DataFrame:
        """
        获取实时行情，支持批量和重试。

        Parameters
        ----------
        codes : list of str
            股票代码列表，如 ['sh600519', 'sz000858']
        max_retries : int
            单批次失败后的最大重试次数
        batch_size : int
            每批次请求的股票数量（腾讯接口限制约80个，留余量取60）

        Returns
        -------
        pd.DataFrame
            行情数据，code_key 为原始代码（如 sh600519）
        """
        if isinstance(codes, str):
            codes = [codes]

        all_quotes: List[Dict[str, Any]] = []

        for i in range(0, len(codes), batch_size):
            batch = codes[i:i + batch_size]
            url = cls.BASE_URL + ','.join(batch)
            quotes = None
            for attempt in range(max_retries + 1):
                try:
                    resp = requests.get(url, timeout=15)
                    resp.encoding = 'gbk'
                    quotes = cls._parse(resp.text)
                    if quotes:
                        break
                except Exception as e:
                    logger.warning(f"批次 {i // batch_size + 1} 第 {attempt + 1} 次请求失败: {e}")
                    if attempt < max_retries:
                        time.sleep(0.5 * (attempt + 1))
                    else:
                        logger.error(f"批次 {i // batch_size + 1} 最终失败，跳过")

            if quotes:
                all_quotes.extend(quotes)

        df = pd.DataFrame(all_quotes)
        if df.empty:
            logger.warning("未获取到任何行情数据")
            return df

        for col in cls.NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    @classmethod
    def _parse(cls, text: str) -> List[Dict[str, Any]]:
        """解析腾讯返回的原始文本"""
        quotes = []
        for line in text.strip().split(';'):
            line = line.strip()
            if not line or 'v_' not in line:
                continue
            parts = line.split('="')
            if len(parts) != 2:
                continue
            code_key = parts[0].replace('v_', '')
            values = parts[1].rstrip('"').split('~')
            quote = {'code_key': code_key}
            for idx, field in cls.FIELDS.items():
                if idx < len(values):
                    quote[field] = values[idx]
            quotes.append(quote)
        return quotes


# -----------------------------------------------------------
# 历史K线
# -----------------------------------------------------------

class TencentHistoryAPI:
    """腾讯历史K线API封装"""

    BASE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

    @classmethod
    def get_kline(cls, symbol: str, start_date: str, end_date: str,
                  period: str = 'day', adjust: str = 'qfq',
                  max_retries: int = 2) -> Optional[pd.DataFrame]:
        """
        获取单只股票历史K线。

        Parameters
        ----------
        symbol : str
            如 'sh600519'
        start_date, end_date : str
            'YYYY-MM-DD'
        period : str
            'day' / 'week' / 'month'
        adjust : str
            'qfq'(前复权) / 'bfq'(不复权)
        max_retries : int
            失败重试次数

        Returns
        -------
        pd.DataFrame | None
            columns: date, open, close, low, high, volume, amount, symbol
        """
        adj_flag = '640' if adjust == 'qfq' else '0'
        url = f"{cls.BASE_URL}?param={symbol},{period},{start_date},{end_date},{adj_flag},{adjust}"

        for attempt in range(max_retries + 1):
            try:
                resp = requests.get(url, timeout=20)
                data = resp.json()

                if data.get('code') != 0:
                    logger.warning(f"[{symbol}] API 返回 code={data.get('code')}: {data.get('msg', '')}")
                    if attempt < max_retries:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    return None

                if not data.get('data') or symbol not in data['data']:
                    logger.warning(f"[{symbol}] 无返回数据")
                    return None

                key = f"{adjust}{period}"  # qfqday / day
                klines = data['data'][symbol].get(key, [])
                if not klines:
                    # 有时 key 会不一样，兜底
                    for k in data['data'][symbol]:
                        if k.endswith(period) and len(data['data'][symbol][k]) > 0:
                            klines = data['data'][symbol][k]
                            break

                if not klines:
                    logger.warning(f"[{symbol}] 无K线数据")
                    return None

                # 数据格式可能为 6 列或 7 列
                # 6 列: [date, open, close, low, high, volume]
                # 7 列: [date, open, close, low, high, volume, amount]
                # 注意：某些行第7个元素可能是分红信息字典，需要截断
                first = klines[0]
                if not isinstance(first, (list, tuple)) or len(first) < 5:
                    logger.error(f"[{symbol}] K线数据格式异常: {first}")
                    return None

                # 取标准列数（最多7列），并截断每行数据到同样长度
                base_cols = ['date', 'open', 'close', 'low', 'high', 'volume', 'amount']
                n_cols = min(len(first), len(base_cols))
                # 处理某些行最后一个元素是字典的情况
                cleaned = [row[:n_cols] for row in klines if isinstance(row, (list, tuple))]
                columns = base_cols[:n_cols]
                df = pd.DataFrame(cleaned, columns=columns)
                df['date'] = pd.to_datetime(df['date'])
                for col in ['open', 'close', 'low', 'high', 'volume', 'amount']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # 剔除异常值
                before = len(df)
                df = df.dropna(subset=['date', 'open', 'close', 'low', 'high'])
                after = len(df)
                if after < before:
                    logger.info(f"[{symbol}] 剔除 {before - after} 条异常K线")

                df['symbol'] = symbol
                return df.sort_values('date').reset_index(drop=True)

            except Exception as e:
                logger.warning(f"[{symbol}] 第 {attempt + 1} 次请求异常: {e}")
                if attempt < max_retries:
                    time.sleep(0.5 * (attempt + 1))
                else:
                    logger.error(f"[{symbol}] 最终获取失败")
                    return None

        return None

    @classmethod
    def get_multi_kline(cls, symbols: List[str], start_date: str, end_date: str,
                        period: str = 'day', adjust: str = 'qfq',
                        delay: float = 0.3, max_retries: int = 2) -> Dict[str, pd.DataFrame]:
        """
        批量获取K线。

        Returns
        -------
        dict
            {symbol: DataFrame}
        """
        results = {}
        for symbol in symbols:
            df = cls.get_kline(symbol, start_date, end_date, period, adjust, max_retries)
            if df is not None and len(df) > 0:
                results[symbol] = df
                logger.info(f"[{symbol}] 获取 {len(df)} 条K线 ({df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()})")
            else:
                logger.warning(f"[{symbol}] 获取失败")
            if delay > 0:
                time.sleep(delay)
        return results

    @classmethod
    def get_nyear_data(cls, symbol: str, years: int = 3,
                       period: str = 'day', adjust: str = 'qfq',
                       delay: float = 0.2) -> Optional[pd.DataFrame]:
        """
        获取多年数据（直接拉取完整区间，避免按年分块的闰年/交易日问题）。
        """
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years + 30)  # 多留30天余量
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        df = cls.get_kline(symbol, start_str, end_str, period, adjust)
        if delay > 0:
            time.sleep(delay)
        return df
