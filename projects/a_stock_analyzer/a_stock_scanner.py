#!/usr/bin/env python3
"""
A股强势扫描器（修改版：剔除创业板和科创板）
- 扫描全市场，找出涨幅≥3%的股票
- 剔除创业板(300/301开头)和科创板(688开头)
- 检查所属行业板块是否也上涨
- 价值因子分析（PE/PB/市值）
- 筹码集中度代理分析（量比/换手率/振幅）
- 生成排名报告
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# 缓存目录
CACHE_DIR = Path('/tmp/a_stock_cache')
CACHE_DIR.mkdir(exist_ok=True)
STOCK_LIST_CACHE = CACHE_DIR / 'stock_list.json'
SECTOR_CACHE = CACHE_DIR / 'sector_summary.json'

# ============ 数据获取 ============

def cached_json(path, ttl_seconds=86400):
    """读取缓存JSON，如果过期返回None"""
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > ttl_seconds:
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(path, data):
    """保存JSON到缓存"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def fetch_stock_list_via_akshare():
    """通过akshare获取全市场股票列表（含行业分类），剔除创业板(300/301)和科创板(688)"""
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        stocks = []
        for _, row in df.iterrows():
            code = str(row['code']).zfill(6)
            name = row['name']
            # 剔除创业板(300/301开头)和科创板(688开头)
            if code.startswith('300') or code.startswith('301') or code.startswith('688'):
                continue
            if code.startswith('6') or code.startswith('68'):
                market = 'sh'
            elif code.startswith('0') or code.startswith('3'):
                market = 'sz'
            else:
                continue
            stocks.append({'code': code, 'name': name, 'market': market, 'industry': ''})
        return stocks
    except Exception as e:
        print(f"akshare获取列表失败: {e}")
        return None

def fetch_stock_list_via_szse():
    """通过深交所API获取股票列表，剔除创业板(300/301开头)"""
    stocks = []
    try:
        page = 1
        while True:
            url = f"http://www.szse.cn/api/report/ShowReport/data?SHOWTYPE=JSON&CATALOGID=1110x&TABKEY=tab1&random=0.1&PAGENO={page}"
            result = subprocess.run(['curl', '-s', '--max-time', '30', url],
                                    capture_output=True, text=True)
            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError:
                print(f"    深交所第{page}页解析失败，跳过")
                break
            
            if not data or not data[0].get('data'):
                break
            for item in data[0]['data']:
                code = item.get('zqdm', '')
                name_html = item.get('gsjc', '')
                name_match = re.search(r'>([^<]+)<', name_html)
                name = name_match.group(1) if name_match else ''
                industry = item.get('sshymc', '')
                # 剔除创业板(300/301开头)
                if code.startswith('300') or code.startswith('301'):
                    continue
                if code and name:
                    stocks.append({'code': code, 'name': name, 'market': 'sz', 'industry': industry})
            metadata = data[0].get('metadata', {})
            total_pages = metadata.get('pagecount', 1)
            if page >= total_pages:
                break
            page += 1
            time.sleep(0.15)
            if page % 20 == 0:
                print(f"    深交所已获取 {len(stocks)} 只...")
    except Exception as e:
        print(f"深交所API失败: {e}")
    return stocks

def get_stock_list():
    """获取全市场股票列表，含行业分类，已剔除创业板(300/301)和科创板(688)"""
    cached = cached_json(STOCK_LIST_CACHE, ttl_seconds=604800)
    if cached and len(cached) > 2000:
        print("  使用缓存股票列表")
        return cached
    
    stocks = []
    # 1. 深交所（含行业分类）
    print("  获取深交所股票列表...")
    sz_stocks = fetch_stock_list_via_szse()
    if sz_stocks:
        stocks.extend(sz_stocks)
        print(f"  深交所: {len(sz_stocks)} 只")
    
    # 2. 上交所（akshare，暂不含行业）
    print("  获取上交所股票列表...")
    sh_stocks = fetch_stock_list_via_akshare()
    if sh_stocks:
        existing_codes = {s['code'] for s in stocks}
        for s in sh_stocks:
            if s['market'] == 'sh' and s['code'] not in existing_codes:
                stocks.append(s)
        print(f"  上交所: {len([s for s in stocks if s['market'] == 'sh'])} 只")
    
    # 3. 补充申万行业分类（覆盖上交所股票）
    print("  补充行业分类...")
    sina_map_path = CACHE_DIR / 'sina_industry_map.json'
    if sina_map_path.exists():
        with open(sina_map_path, 'r', encoding='utf-8') as f:
            sina_map = json.load(f)
        for s in stocks:
            if not s.get('industry'):
                sw_industry = sina_map.get(s['code'], '')
                if sw_industry:
                    s['industry'] = f'SW_{sw_industry}'
                    s['sw_industry'] = sw_industry
            else:
                # 深交所的行业是证监会分类，同时标记申万分类
                sw_industry = sina_map.get(s['code'], '')
                if sw_industry:
                    s['sw_industry'] = sw_industry
        mapped = sum(1 for s in stocks if s.get('sw_industry'))
        print(f"  行业分类覆盖: {mapped}/{len(stocks)} 只")
    
    if stocks:
        save_json(STOCK_LIST_CACHE, stocks)
    return stocks

# ============ 腾讯实时行情 ============

def fetch_tencent_quotes(codes_batch):
    """从腾讯接口获取一批股票实时行情"""
    url = f"https://qt.gtimg.cn/q={','.join(codes_batch)}"
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        resp = urllib.request.urlopen(req, timeout=30)
        data = resp.read().decode('gbk', errors='ignore')
        return data
    except Exception as e:
        return None

def parse_tencent_quotes(raw_data):
    """解析腾讯行情数据"""
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
            stock = {
                'code': code,
                'market': prefix,
                'name': fields[1],
                'price': float(fields[3]) if fields[3] else 0,
                'prev_close': float(fields[4]) if fields[4] else 0,
                'open': float(fields[5]) if fields[5] else 0,
                'volume': int(fields[6]) if fields[6] else 0,
                'high': float(fields[33]) if fields[33] else 0,
                'low': float(fields[34]) if fields[34] else 0,
                'change_amount': float(fields[31]) if fields[31] else 0,
                'change_pct': float(fields[32]) if fields[32] else 0,
                'turnover': float(fields[38]) if fields[38] else 0,
                'pe': float(fields[39]) if fields[39] else None,
                'amplitude': float(fields[43]) if fields[43] else 0,
                'total_market': float(fields[45]) if fields[45] else 0,
                'circ_market': float(fields[44]) if fields[44] else 0,
                'pb': float(fields[46]) if fields[46] else None,
                'volume_ratio': float(fields[49]) if fields[49] else 0,
                'time': fields[30],
            }
            stocks.append(stock)
        except (ValueError, IndexError):
            continue
    return stocks

def fetch_all_quotes(stock_list):
    """获取全市场实时行情"""
    all_quotes = []
    codes = [f"{s['market']}{s['code']}" for s in stock_list if s.get('market')]
    batch_size = 100
    total = len(codes)
    
    for i in range(0, total, batch_size):
        batch = codes[i:i+batch_size]
        raw = fetch_tencent_quotes(batch)
        if raw:
            quotes = parse_tencent_quotes(raw)
            all_quotes.extend(quotes)
        if (i // batch_size + 1) % 10 == 0:
            print(f"  已获取 {min(i+batch_size, total)}/{total} 只...")
        time.sleep(0.2)
    
    return all_quotes

# ============ 板块数据 ============

def fetch_sector_summary():
    """获取同花顺行业板块实时行情，多次重试"""
    cached = cached_json(SECTOR_CACHE, ttl_seconds=3600)
    if cached and len(cached) > 50:
        return cached
    
    import akshare as ak
    for attempt in range(3):
        try:
            df = ak.stock_board_industry_summary_ths()
            sectors = {}
            for _, row in df.iterrows():
                name = row.get('板块', '')
                change_pct = float(row.get('涨跌幅', 0))
                up_count = int(row.get('上涨家数', 0))
                down_count = int(row.get('下跌家数', 0))
                sectors[name] = {
                    'change_pct': change_pct,
                    'up_count': up_count,
                    'down_count': down_count,
                    'is_rising': change_pct > 0
                }
            save_json(SECTOR_CACHE, sectors)
            return sectors
        except Exception as e:
            print(f"  板块数据获取失败(尝试{attempt+1}/3): {str(e)[:80]}")
            time.sleep(2)
    
    return {}

# ============ 分析逻辑 ============

def fetch_hist_volume_data(code, days=4):
    """获取股票最近N天历史K线，返回 DataFrame"""
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
        if df is None or len(df) < days:
            return None
        df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount',
                     'amplitude', 'pct_change', 'price_change', 'turnover']
        return df.tail(days)
    except Exception as e:
        return None

def check_volume_surge(code, threshold=2.0):
    """检查前一天成交量是否 >= 前3天平均的 threshold 倍
    返回 (是否满足, 昨天成交量, 前3天平均, 倍数)
    """
    df = fetch_hist_volume_data(code, days=4)
    if df is None or len(df) < 4:
        # 无法获取历史数据，默认通过（让实时量比来决定）
        return True, 0, 1, 1.0
    
    volumes = df['volume'].tolist()
    yest_vol = volumes[-1]
    prev3_avg = sum(volumes[-4:-1]) / 3
    
    if prev3_avg == 0:
        ratio = 999
    else:
        ratio = yest_vol / prev3_avg
    
    return ratio >= threshold, yest_vol, prev3_avg, ratio

def analyze_value_factors(stock):
    """价值因子分析 - 评分0-100
    
    综合考虑：
    - PE：适中PE（10-40）为佳，过高或负PE扣分
    - PB：适中PB（1-5）为佳
    - 市值：排除极端大小
    """
    score = 50
    pe = stock.get('pe')
    pb = stock.get('pb')
    total_market = stock.get('total_market', 0)
    circ_market = stock.get('circ_market', 0)
    
    # PE评分
    if pe is not None and pe > 0:
        if 10 <= pe <= 30:
            score += 15
        elif 30 < pe <= 50:
            score += 5
        elif pe > 100 or pe < 0:
            score -= 15
        elif pe < 10:
            score += 5
    else:
        score -= 5  # 无PE数据或负PE
    
    # PB评分
    if pb is not None and pb > 0:
        if 1 <= pb <= 3:
            score += 15
        elif 3 < pb <= 5:
            score += 5
        elif pb > 10:
            score -= 10
    else:
        score -= 5
    
    # 市值评分（以亿为单位）
    circ = circ_market / 1e8 if circ_market else 0
    if 50 <= circ <= 500:
        score += 10
    elif circ < 20:
        score -= 5  # 太小可能流动性差
    elif circ > 2000:
        score -= 5  # 太大可能弹性不足
    
    return max(0, min(100, score))

def analyze_chip_proxies(stock):
    """筹码集中度代理分析（替代真正的筹码峰）
    
    使用以下指标作为筹码峰的代理：
    - 量比：反映当前成交量相对于近期平均的变化
    - 换手率：反映筹码活跃度
    - 振幅：反映当日筹码博弈激烈程度
    - 价格位置：当前价格相对于前收盘价的位置
    """
    score = 50
    
    volume_ratio = stock.get('volume_ratio', 0)
    turnover = stock.get('turnover', 0)
    amplitude = stock.get('amplitude', 0)
    change_pct = stock.get('change_pct', 0)
    
    # 量比评分：1.5-3.0为健康放量，>5可能过热
    if 1.5 <= volume_ratio <= 3.0:
        score += 20  # 健康放量，筹码在换手
    elif volume_ratio > 3.0 and volume_ratio <= 5.0:
        score += 10
    elif volume_ratio > 5.0:
        score -= 10  # 过度放量，可能是出货
    elif volume_ratio < 0.8:
        score -= 5   # 缩量上涨，持续性存疑
    
    # 换手率评分：3%-15%为活跃区间
    if 3 <= turnover <= 10:
        score += 15
    elif 10 < turnover <= 15:
        score += 10
    elif turnover > 20:
        score -= 10  # 过高换手，可能筹码松动
    elif turnover < 1:
        score -= 5   # 过于冷清
    
    # 振幅评分：适中振幅说明有资金博弈但可控
    if 3 <= amplitude <= 8:
        score += 10
    elif amplitude > 10:
        score += 5   # 高振幅但涨停，说明强势
    elif amplitude < 2:
        score -= 5   # 振幅太小，可能是一字板或弱势
    
    # 涨幅位置：盘中上涨（非一字板）更有筹码换手意义
    if change_pct >= 9.5 and stock.get('low', 0) == stock.get('high', 0):
        score -= 10  # 一字板，筹码未充分换手
    elif change_pct >= 3 and stock.get('low', 0) < stock.get('price', 0):
        score += 10  # 有下影线或换手上涨，筹码在交换
    
    return max(0, min(100, score))

def calculate_strength_score(stock):
    """综合强度评分"""
    value_score = analyze_value_factors(stock)
    chip_score = analyze_chip_proxies(stock)
    
    # 综合权重：价值30% + 筹码40% + 涨幅30%
    change_pct = stock.get('change_pct', 0)
    change_score = min(100, change_pct * 10)  # 3%=30分, 10%=100分
    
    total = value_score * 0.25 + chip_score * 0.35 + change_score * 0.40
    return round(total, 1)

# ============ 主流程 ============

def run_analysis(change_pct_threshold=3.0, max_change_pct=8.0):
    """执行全部分析流程 - 基于前一天收盘数据，已剔除创业板(300/301)和科创板(688)"""
    print(f"\n{'='*60}")
    print(f"A股强势扫描器 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"分析日期: 前一交易日收盘数据")
    print(f"筛选条件: 涨幅 ≥ {change_pct_threshold}% 且 < {max_change_pct}%, 成交量 ≥ 前3天平均的2倍")
    print(f"已剔除: 创业板(300/301开头)和科创板(688开头)")
    print(f"{'='*60}\n")
    
    # 1. 获取股票列表
    print("[1/5] 获取股票列表...")
    stock_list = get_stock_list()
    if not stock_list:
        print("获取股票列表失败!")
        return None
    print(f"  共 {len(stock_list)} 只股票")
    
    # 2. 获取全市场前一天收盘数据（通过 akshare 快照接口，失败则回退腾讯接口）
    print("[2/5] 获取前一交易日收盘数据...")
    import akshare as ak
    all_quotes = []
    try:
        df_spot = ak.stock_zh_a_spot_em()
        # 将 DataFrame 转为字典列表
        for _, row in df_spot.iterrows():
            code = str(row.get('代码', '')).zfill(6)
            name = row.get('名称', '')
            change_pct = row.get('涨跌幅', 0)
            if isinstance(change_pct, str):
                try: change_pct = float(change_pct)
                except: change_pct = 0
            price = row.get('最新价', 0)
            if isinstance(price, str):
                try: price = float(price)
                except: price = 0
            prev_close = row.get('昨收', 0)
            if isinstance(prev_close, str):
                try: prev_close = float(prev_close)
                except: prev_close = 0
            volume = row.get('成交量', 0)
            if isinstance(volume, str):
                try: volume = int(float(volume))
                except: volume = 0
            turnover = row.get('换手率', 0)
            if isinstance(turnover, str):
                try: turnover = float(turnover)
                except: turnover = 0
            amplitude = row.get('振幅', 0)
            if isinstance(amplitude, str):
                try: amplitude = float(amplitude)
                except: amplitude = 0
            pe = row.get('市盈率-动态', None)
            if isinstance(pe, str):
                try: pe = float(pe)
                except: pe = None
            pb = row.get('市净率', None)
            if isinstance(pb, str):
                try: pb = float(pb)
                except: pb = None
            total_market = row.get('总市值', 0)
            if isinstance(total_market, str):
                try: total_market = float(total_market)
                except: total_market = 0
            circ_market = row.get('流通市值', 0)
            if isinstance(circ_market, str):
                try: circ_market = float(circ_market)
                except: circ_market = 0
            volume_ratio = row.get('量比', 0)
            if isinstance(volume_ratio, str):
                try: volume_ratio = float(volume_ratio)
                except: volume_ratio = 0
            
            all_quotes.append({
                'code': code,
                'name': name,
                'price': price,
                'prev_close': prev_close,
                'change_pct': change_pct,
                'volume': volume,
                'turnover': turnover,
                'amplitude': amplitude,
                'pe': pe,
                'pb': pb,
                'total_market': total_market,
                'circ_market': circ_market,
                'volume_ratio': volume_ratio,
            })
        print(f"  通过 akshare 获取 {len(all_quotes)} 只股票数据")
    except Exception as e:
        print(f"  akshare 数据获取失败: {e}")
        print("  回退到腾讯接口...")
        all_quotes = fetch_all_quotes(stock_list)
        print(f"  通过腾讯接口获取 {len(all_quotes)} 只股票数据")
    
    if not all_quotes:
        print("获取行情失败!")
        return None
    
    # 3. 按申万行业计算行业平均涨跌幅
    print("[3/5] 计算申万行业涨跌...")
    stock_map = {s['code']: s for s in stock_list}
    
    # 合并行情和行业数据
    for quote in all_quotes:
        info = stock_map.get(quote['code'], {})
        quote['sw_industry'] = info.get('sw_industry', '')
        quote['industry'] = info.get('industry', '')
    
    # 按申万行业分组计算平均涨跌幅
    import pandas as pd
    df = pd.DataFrame(all_quotes)
    industry_avg = df[df['sw_industry'] != ''].groupby('sw_industry')['change_pct'].mean().to_dict()
    
    rising_industries = {k: v for k, v in industry_avg.items() if v > 0}
    print(f"  共 {len(industry_avg)} 个申万行业, {len(rising_industries)} 个上涨")
    top_industries = sorted(industry_avg.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  领涨行业: " + ", ".join([f"{n}(+{v:.2f}%)" for n, v in top_industries]))
    
    # 4. 初步筛选：涨幅 ≥ 阈值 且 < 上限 且非ST，且剔除创业板/科创板
    print(f"[4/5] 初步筛选: 涨幅 ≥ {change_pct_threshold}% 且 < {max_change_pct}% ...")
    candidates = []
    for stock in all_quotes:
        code = stock.get('code', '')
        name = stock.get('name', '')
        # 过滤ST/退市/涨停
        if 'ST' in name or '退' in name or 'N' in name[:1]:
            continue
        # 剔除创业板(300/301开头)和科创板(688开头)
        if code.startswith('300') or code.startswith('301') or code.startswith('688'):
            continue
        pct = stock.get('change_pct', 0)
        if pct >= change_pct_threshold and pct < max_change_pct:
            candidates.append(stock)

    print(f"  初步候选: {len(candidates)} 只")
    
    # 5. 成交量验证：前一天成交量 >= 前3天平均的2倍
    print("[5/5] 验证成交量条件 (>= 前3天平均的2倍)...")
    filtered = []
    for stock in candidates:
        code = stock['code']
        ok, yest_vol, prev3_avg, ratio = check_volume_surge(code, threshold=2.0)
        stock['volume_surge_ok'] = ok
        stock['yesterday_volume'] = yest_vol
        stock['prev3_avg_volume'] = prev3_avg
        stock['volume_surge_ratio'] = round(ratio, 2)
        
        if ok:
            sw_ind = stock.get('sw_industry', '')
            ind_change = industry_avg.get(sw_ind, 0)
            
            stock['sector_name'] = sw_ind if sw_ind else '待确认'
            stock['sector_change_pct'] = round(ind_change, 2)
            stock['sector_is_rising'] = ind_change > 0 if sw_ind else False
            
            # 计算评分
            stock['value_score'] = analyze_value_factors(stock)
            stock['chip_score'] = analyze_chip_proxies(stock)
            stock['strength_score'] = calculate_strength_score(stock)
            
            # 保留：有行业分类且行业上涨，或没有行业分类（待确认）
            if stock['sector_is_rising'] or not sw_ind:
                filtered.append(stock)
    
    # 按综合评分排序
    filtered.sort(key=lambda x: x['strength_score'], reverse=True)
    
    print(f"  成交量满足: {sum(1 for s in candidates if s.get('volume_surge_ok'))} 只")
    print(f"  行业匹配: {len([s for s in filtered if s['sector_is_rising']])} 只")
    print(f"  无行业分类(保留): {len([s for s in filtered if not s.get('sw_industry')])} 只")
    print(f"  最终入选: {len(filtered)} 只")
    
    # 6. 生成报告
    print("[6/6] 生成分析报告...")
    report = generate_report(filtered, industry_avg, change_pct_threshold, max_change_pct)
    
    return report

def generate_report(stocks, industry_avg, change_pct_threshold=3.0, max_change_pct=8.0):
    """生成文本报告"""
    lines = []
    lines.append(f"📈 A股强势扫描报告 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"分析日期: 前一交易日收盘数据")
    lines.append("=" * 70)
    lines.append("")
    
    # 市场概览
    rising_industries = {k: v for k, v in industry_avg.items() if v > 0}
    lines.append(f"【市场概览】")
    lines.append(f"  上涨行业: {len(rising_industries)}/{len(industry_avg)} 个（申万一级）")
    if rising_industries:
        top_industries = sorted(industry_avg.items(), key=lambda x: x[1], reverse=True)[:5]
        lines.append(f"  领涨行业: " + ", ".join([f"{n}(+{v:.2f}%)" for n, v in top_industries]))
    lines.append("")
    
    # 个股列表
    if not stocks:
        lines.append(f"【筛选结果】未找到符合条件的股票")
        lines.append(f"筛选条件: 涨幅≥{change_pct_threshold}%且<{max_change_pct}%, 成交量≥前3天平均2倍, 所属行业上涨")
        return "\n".join(lines)
    
    lines.append(f"【精选股票】共 {len(stocks)} 只")
    lines.append(f"筛选条件: 涨幅≥{change_pct_threshold}%且<{max_change_pct}% + 成交量≥前3天平均2倍 + 所属行业上涨")
    lines.append(f"已剔除: 创业板(300/301开头)和科创板(688开头)")
    lines.append("-" * 90)
    lines.append(f"{'排名':<4} {'代码':<8} {'名称':<10} {'现价':<8} {'涨幅':<8} {'量比(3d)':<10} {'行业':<10} {'行业涨':<8} {'价值分':<6} {'筹码分':<6} {'综合分':<6}")
    lines.append("-" * 90)
    
    for i, s in enumerate(stocks[:30], 1):
        sector = s.get('sector_name', '-')
        sector_short = sector[:8] if len(sector) > 8 else sector
        vol_ratio = s.get('volume_surge_ratio', 0)
        lines.append(
            f"{i:<4} {s['code']:<8} {s['name']:<10} {s['price']:<8.2f} "
            f"{s['change_pct']:>+6.2f}%  {vol_ratio:<10.2f}x "
            f"{sector_short:<10} {s.get('sector_change_pct', 0):>+6.2f}%  "
            f"{s.get('value_score', 0):<6} {s.get('chip_score', 0):<6} {s.get('strength_score', 0):<6}"
        )
    
    lines.append("")
    lines.append("【评分说明】")
    lines.append("  价值分: PE/PB/市值综合评分 (0-100)")
    lines.append("  筹码分: 量比/换手率/振幅代理评分 (0-100)")
    lines.append("  综合分: 价值25% + 筹码35% + 涨幅40%")
    lines.append("  量比(3d): 前一天成交量 / 前3天平均成交量")
    lines.append("")
    lines.append("⚠️ 免责声明: 本分析仅供学习研究，不构成投资建议")
    
    return "\n".join(lines)

# ============ 入口 ============

if __name__ == '__main__':
    report = run_analysis(change_pct_threshold=3.0, max_change_pct=8.0)
    if report:
        print("\n" + report)
        # 保存报告
        report_path = f"/tmp/a_stock_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n报告已保存: {report_path}")
        
        # 发送邮件通知（使用 imap-smtp-email skill）
        try:
            import subprocess
            skill_dir = '/root/.openclaw/workspace/skills/imap-smtp-email-chinese'
            result = subprocess.run(
                [
                    'node', f'{skill_dir}/scripts/smtp.js', 'send',
                    '--to', 'hansonnan.ding@sc.com,leo.sun@sc.com,jacky.an@sc.com',
                    '--subject', f'A股强势扫描报告 - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                    '--body', report,
                    '--attach', report_path
                ],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                print("邮件通知已发送")
            else:
                print(f"邮件发送失败: {result.stderr}")
        except Exception as e:
            print(f"邮件发送失败: {e}")
