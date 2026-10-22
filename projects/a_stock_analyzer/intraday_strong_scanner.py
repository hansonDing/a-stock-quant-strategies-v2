#!/usr/bin/env python3
"""
A股盘中强势扫描器（14:45 尾盘时段）
- 基于当天盘中实时涨幅 + 当日成交量（vs 5日均量）
- 初筛用腾讯实时字段（涨幅/量比/换手/成交额），复筛对候选拉5日均量精确验证
- 评分: 涨幅40% + 量能40% + 换手20%
- 剔除创业板(300/301)/科创板(688)/ST/退市/新股/一字板，与08:30版本一致
"""

import json
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path

CACHE_DIR = Path('/tmp/a_stock_cache')
CACHE_DIR.mkdir(exist_ok=True)
STOCK_LIST_CACHE = CACHE_DIR / 'stock_list.json'

# ============ 筛选参数 ============
CHANGE_MIN = 3.0        # 当日涨幅下限（%）
CHANGE_MAX = 7.0        # 当日涨幅上限（%），排除即将涨停/已涨停
TURNOVER_MIN = 3.0      # 换手率下限（%）
TURNOVER_MAX = 15.0     # 换手率上限（%），过高可能筹码松动
MIN_AMOUNT = 2.0        # 成交额下限（亿元）
VOL_RATIO_MIN = 1.7     # 当日成交量 / 5日均量 下限（14:45累计量，未修正时段，偏严）
TOP_N = 20              # 报告输出数量

# ============ 腾讯实时行情 ============

def fetch_tencent_quotes(codes_batch):
    url = f"https://qt.gtimg.cn/q={','.join(codes_batch)}"
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.read().decode('gbk', errors='ignore')
    except Exception:
        return None

def parse_tencent_quotes(raw_data):
    stocks = []
    for line in raw_data.strip().split('\n'):
        if not line.strip():
            continue
        m = re.search(r'v_(sh|sz)(\d+)="([^"]*)"', line)
        if not m:
            continue
        prefix, code, content = m.group(1), m.group(2), m.group(3)
        fields = content.split('~')
        if len(fields) < 50:
            continue
        try:
            stocks.append({
                'code': code,
                'market': prefix,
                'name': fields[1],
                'price': float(fields[3]) if fields[3] else 0,
                'prev_close': float(fields[4]) if fields[4] else 0,
                'open': float(fields[5]) if fields[5] else 0,
                'volume': int(fields[6]) if fields[6] else 0,      # 手
                'amount': float(fields[37]) if fields[37] else 0,  # 万元
                'time': fields[30],
                'change_pct': float(fields[32]) if fields[32] else 0,
                'high': float(fields[33]) if fields[33] else 0,
                'low': float(fields[34]) if fields[34] else 0,
                'turnover': float(fields[38]) if fields[38] else 0,
                'amplitude': float(fields[43]) if fields[43] else 0,
                'circ_market': float(fields[44]) if fields[44] else 0,
                'volume_ratio': float(fields[49]) if fields[49] else 0,
            })
        except (ValueError, IndexError):
            continue
    return stocks

def fetch_all_quotes(stock_list):
    all_quotes = []
    codes = [f"{s['market']}{s['code']}" for s in stock_list if s.get('market')]
    batch_size = 100
    total = len(codes)
    for i in range(0, total, batch_size):
        raw = fetch_tencent_quotes(codes[i:i+batch_size])
        if raw:
            all_quotes.extend(parse_tencent_quotes(raw))
        if (i // batch_size + 1) % 20 == 0:
            print(f"  已获取 {min(i+batch_size, total)}/{total} 只...")
        time.sleep(0.15)
    return all_quotes

# ============ 历史均量（腾讯K线优先，akshare兜底，各重试2次） ============

def _avg_from_rows(vols):
    if not vols or len(vols) < 5:
        return None
    return sum(vols[-5:]) / 5

def _get_avg_volume_tencent(code, market):
    """腾讯历史K线接口，返回近5日均量（手）。失败返回None"""
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={market}{code},day,,,10,qfq"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=15)
    data = json.loads(resp.read().decode('utf-8'))
    klines = (data.get('data', {}).get(f'{market}{code}', {})
              .get('qfqday') or data.get('data', {}).get(f'{market}{code}', {}).get('day') or [])
    vols = [float(k[5]) for k in klines if len(k) > 5]
    return _avg_from_rows(vols)

def _get_avg_volume_akshare(code):
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
        if df is None or len(df) < 5:
            return None
        col = '成交量' if '成交量' in df.columns else df.columns[5]
        return float(df[col].tail(5).mean())
    except Exception:
        return None

def get_avg_volume_5d(code, market):
    """近5日平均成交量（手）。腾讯接口优先，akshare兜底，各重试2次"""
    for attempt in range(2):
        try:
            v = _get_avg_volume_tencent(code, market)
            if v:
                return v
        except Exception:
            time.sleep(0.5)
    for attempt in range(2):
        try:
            v = _get_avg_volume_akshare(code)
            if v:
                return v
        except Exception:
            time.sleep(0.5)
    return None

# ============ 评分 ============

def score_change(pct):
    """涨幅得分 0-100：3%→30分，7%+→100分，线性"""
    if pct < CHANGE_MIN:
        return 0
    s = 30 + (pct - CHANGE_MIN) / (CHANGE_MAX - CHANGE_MIN) * 70
    return min(100, s)

def score_volume(ratio):
    """量能得分 0-100：1.7x→40分，4x+→100分，线性"""
    if ratio < 1.0:
        return 0
    s = 40 + (ratio - VOL_RATIO_MIN) / (4.0 - VOL_RATIO_MIN) * 60
    return max(0, min(100, s))

def score_turnover(t):
    """换手得分 0-100：3-10%区间最优"""
    if 3 <= t <= 10:
        return 100
    if 1 <= t < 3:
        return 40 + (t - 1) / 2 * 60
    if 10 < t <= 15:
        return 100 - (t - 10) / 5 * 40
    return 0

# ============ 主流程 ============

def run():
    now = datetime.now()
    print(f"\n{'='*60}")
    print(f"A股盘中强势扫描 - {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据: 当日盘中实时（14:45尾盘时段，当日量未修正）")
    print(f"筛选: 涨幅{CHANGE_MIN}%~{CHANGE_MAX}% + 量比≥{VOL_RATIO_MIN}x(5日均量) + 换手{TURNOVER_MIN}%~{TURNOVER_MAX}% + 成交额≥{MIN_AMOUNT}亿")
    print(f"剔除: 创业板/科创板/ST/退市/新股/一字板")
    print(f"{'='*60}\n")

    # 1. 股票列表（复用08:30版本缓存，已剔除创业板/科创板）
    print("[1/4] 获取股票列表...")
    if STOCK_LIST_CACHE.exists():
        stock_list = json.loads(STOCK_LIST_CACHE.read_text(encoding='utf-8'))
        print(f"  缓存 {len(stock_list)} 只（主板）")
    else:
        print("  无缓存，请先运行 a_stock_scanner.py 生成股票列表")
        return None

    # 2. 全市场实时行情
    print("[2/4] 拉取全市场实时行情（腾讯接口）...")
    all_quotes = fetch_all_quotes(stock_list)
    print(f"  成功 {len(all_quotes)} 只")
    if not all_quotes:
        print("行情获取失败")
        return None

    # 3. 初筛（纯实时字段，快）
    print("[3/4] 初筛（涨幅/换手/成交额/一字板）...")
    pre = []
    for s in all_quotes:
        code, name = s['code'], s['name']
        if 'ST' in name or '退' in name or name.startswith('N'):
            continue
        pct = s['change_pct']
        if not (CHANGE_MIN <= pct < CHANGE_MAX):
            continue
        if not (TURNOVER_MIN <= s['turnover'] <= TURNOVER_MAX):
            continue
        if s['amount'] < MIN_AMOUNT * 1e4:  # 万元→亿
            continue
        if s['amplitude'] < 1.0:   # 一字板/近似一字：几乎无换手意义
            continue
        pre.append(s)
    print(f"  初筛候选: {len(pre)} 只")

    # 4. 复筛：5日均量精确验证放量
    print("[4/4] 复筛（当日量 vs 5日均量）...")
    filtered = []
    fails = 0
    for s in pre:
        avg5 = get_avg_volume_5d(s['code'], s['market'])
        if avg5 is None or avg5 == 0:
            fails += 1
            continue
        ratio = s['volume'] / avg5
        if ratio >= VOL_RATIO_MIN:
            s['vol_ratio_5d'] = round(ratio, 2)
            c = score_change(s['change_pct'])
            v = score_volume(ratio)
            t = score_turnover(s['turnover'])
            s['change_score'] = round(c, 1)
            s['volume_score'] = round(v, 1)
            s['turnover_score'] = round(t, 1)
            s['strength'] = round(c * 0.4 + v * 0.4 + t * 0.2, 1)
            filtered.append(s)
        time.sleep(0.12)
    print(f"  历史数据失败跳过: {fails} 只")
    print(f"  最终入选: {len(filtered)} 只")

    filtered.sort(key=lambda x: x['strength'], reverse=True)
    return generate_report(filtered, len(all_quotes), len(pre), fails)

def generate_report(stocks, total, pre_cnt, fail_cnt):
    now = datetime.now()
    lines = []
    lines.append(f"📈 A股盘中强势扫描 - {now.strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 78)
    lines.append("")
    lines.append("【计算逻辑】")
    lines.append(f"  涨幅 {CHANGE_MIN}%~{CHANGE_MAX}%（强势未涨停，有参与空间）")
    lines.append(f"  量能：当日成交量 ≥ 近5日均量 × {VOL_RATIO_MIN}（14:45累计量，未做时段修正，标准偏严）")
    lines.append(f"  换手 {TURNOVER_MIN}%~{TURNOVER_MAX}%，成交额 ≥ {MIN_AMOUNT}亿")
    lines.append(f"  评分 = 涨幅×40% + 量能×40% + 换手×20%")
    lines.append(f"  剔除：创业板/科创板/ST/退市/新股/一字板")
    lines.append("")
    lines.append(f"【扫描概况】全市场 {total} 只 → 初筛 {pre_cnt} 只 → 复筛入选 {len(stocks)} 只")
    lines.append("")

    if not stocks:
        lines.append("【结果】无符合条件标的")
        lines.append("⚠️ 免责声明: 本扫描仅供学习研究，不构成投资建议")
        return "\n".join(lines)

    lines.append(f"【推荐 TOP{min(TOP_N, len(stocks))}】")
    lines.append("-" * 88)
    lines.append(f"{'#':<3} {'代码':<7} {'名称':<9} {'现价':<8} {'涨幅%':<8} {'量比5d':<8} {'换手%':<7} {'成交额亿':<8} {'涨幅分':<6} {'量能分':<6} {'综合':<6}")
    lines.append("-" * 88)
    for i, s in enumerate(stocks[:TOP_N], 1):
        lines.append(
            f"{i:<3} {s['code']:<7} {s['name']:<9} {s['price']:<8.2f} "
            f"{s['change_pct']:>+6.2f}  {s['vol_ratio_5d']:<8.2f} {s['turnover']:>6.2f}  "
            f"{s['amount']/1e4:>8.2f}  {s['change_score']:<6} {s['volume_score']:<6} {s['strength']:<6}"
        )
    lines.append("")
    lines.append("⚠️ 免责声明: 本扫描基于盘中数据，仅供学习研究，不构成投资建议")
    return "\n".join(lines)

if __name__ == '__main__':
    report = run()
    if report:
        print("\n" + report)
        rp = f"/tmp/intraday_strong_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        Path(rp).write_text(report, encoding='utf-8')
        print(f"\n报告已保存: {rp}")
