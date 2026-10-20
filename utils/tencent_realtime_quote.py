#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
腾讯财经实时行情接口封装（兼容层）
==================================
已迁移到 utils/api.py 中的 TencentQuoteAPI 类。
本文件保留为兼容层，直接调用公共模块。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.api import TencentQuoteAPI, format_quote  # noqa

if __name__ == '__main__':
    test_codes = ['sh600519', 'sz000858', 'sh601318', 'sz000333', 'sh600036', 'hk00700', 'aapl']
    print(f"正在获取 {len(test_codes)} 只股票实时行情...")
    df = TencentQuoteAPI.get_quotes(test_codes)
    from utils.api import format_quote
    format_quote(df)
