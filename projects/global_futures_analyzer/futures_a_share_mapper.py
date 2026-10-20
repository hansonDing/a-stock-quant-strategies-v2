#!/usr/bin/env python3
"""
国际期货 → A股映射扫描器
- 每天早晨扫描国际期货涨幅最大的3个品种
- 根据期货品种映射到A股相关行业
- 从沪深主板（60/00开头）筛选预期受益股票
- 生成报告并发送邮件
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

# ============ 期货品种 → A股行业映射 ============

FUTURES_TO_A_SHARE_MAP = {
    # 能源 - 石油石化（精准）
    '原油': {'keywords': ['石油', '石化', '油服', '海油', '中海油', '中石油', '中石化', '油气开采'], 'industries': ['石油石化', '油气开采']},
    '燃油': {'keywords': ['石油', '石化', '油服', '海油', '油气'], 'industries': ['石油石化']},
    '柴油': {'keywords': ['石油', '石化', '油服', '海油', '油气'], 'industries': ['石油石化']},
    '重柴油': {'keywords': ['石油', '石化', '油服', '海油', '油气'], 'industries': ['石油石化']},
    '天然气': {'keywords': ['燃气', '天然气'], 'industries': ['燃气']},
    
    # 贵金属
    '黄金': {'keywords': ['黄金', '中金', '山金', '紫金', '贵金'], 'industries': ['贵金属', '黄金']},
    '白银': {'keywords': ['白银', '贵金'], 'industries': ['贵金属']},
    '铂金': {'keywords': ['铂金', '贵金'], 'industries': ['贵金属']},
    '钯金': {'keywords': ['钯金', '贵金'], 'industries': ['贵金属']},
    
    # 工业金属
    '铜': {'keywords': ['铜业', '铜'], 'industries': ['铜', '工业金属']},
    '铝': {'keywords': ['铝业', '铝'], 'industries': ['铝', '工业金属']},
    '锌': {'keywords': ['锌业', '锌'], 'industries': ['锌', '小金属']},
    '镍': {'keywords': ['镍'], 'industries': ['镍', '小金属']},
    '铅': {'keywords': ['铅'], 'industries': ['铅', '小金属']},
    '锡': {'keywords': ['锡'], 'industries': ['锡', '小金属']},
    
    # 农产品
    '大豆': {'keywords': ['大豆', '豆粕', '饲料'], 'industries': ['饲料', '农产品加工']},
    '豆粕': {'keywords': ['豆粕', '饲料'], 'industries': ['饲料']},
    '玉米': {'keywords': ['玉米', '种业', '种植'], 'industries': ['种业', '种植']},
    '小麦': {'keywords': ['小麦', '种业', '种植'], 'industries': ['种业', '种植']},
    '燕麦': {'keywords': ['种业', '种植', '农业'], 'industries': ['种植']},
    '棕榈油': {'keywords': ['棕榈', '油脂'], 'industries': ['食品加工', '油脂']},
    '棉花': {'keywords': ['棉花', '纺织', '棉纺'], 'industries': ['纺织制造']},
    '糖': {'keywords': ['糖业', '白糖', '制糖'], 'industries': ['制糖']},
    
    # 股指/利率
    'A50': {'keywords': ['证券', '银行', '保险', '券商'], 'industries': ['证券', '银行', '保险']},
    '国债': {'keywords': ['银行', '保险'], 'industries': ['银行', '保险']},
    '美债': {'keywords': ['银行', '保险'], 'industries': ['银行', '保险']},
    '利率': {'keywords': ['银行', '保险'], 'industries': ['银行', '保险']},
    
    # 其他
    '橡胶': {'keywords': ['橡胶'], 'industries': ['橡胶']},
    '碳排放': {'keywords': ['环保', '碳'], 'industries': ['环保']},
    '瘦肉猪': {'keywords': ['养殖', '猪'], 'industries': ['养殖']},
}

def match_futures_to_keywords(futures_name):
    """根据期货名称匹配A股关键词"""
    matched = []
    futures_name_lower = futures_name.lower()
    
    for keyword, mapping in FUTURES_TO_A_SHARE_MAP.items():
        if keyword in futures_name:
            matched.append(mapping)
    
    return matched

def extract_commodity_name(name):
    """从期货名称提取商品名，去掉合约月份"""
    # 去掉末尾的月份代码如 2405, 2506, 00Y 等
    name = re.sub(r'\d{4}[A-Z]$', '', name)
    name = re.sub(r'\d{2}[A-Z]$', '', name)
    name = re.sub(r'\d{2,4}$', '', name)
    name = re.sub(r'当月连续|主力合约', '', name)
    return name.strip()

# ============ 数据获取 ============

def fetch_global_futures(max_retries=3, base_delay=2.0):
    """获取国际期货实时行情，带指数退避重试"""
    for attempt in range(max_retries):
        try:
            import akshare as ak
            time.sleep(1.0)  # 先等1秒，避免连续请求
            df = ak.futures_global_spot_em()
            
            # 过滤有交易的合约
            df = df[df['成交量'] > 0].copy()
            
            # 去掉"当月连续"和"主力合约"，保留具体合约
            # 但保留品种代表性
            df['商品名'] = df['名称'].apply(extract_commodity_name)
            
            # 按商品名聚合，取成交量最大的合约代表该品种
            df_sorted = df.sort_values('成交量', ascending=False)
            
            # 去重：每个商品名只保留成交量最大的一个合约
            seen = set()
            unique_futures = []
            for _, row in df_sorted.iterrows():
                commodity = row['商品名']
                if commodity not in seen and len(commodity) > 0:
                    seen.add(commodity)
                    unique_futures.append({
                        'code': row['代码'],
                        'name': row['名称'],
                        'commodity': commodity,
                        'price': row['最新价'],
                        'change_pct': row['涨跌幅'],
                        'change_amount': row['涨跌额'],
                        'volume': row['成交量'],
                        'open': row['今开'],
                        'high': row['最高'],
                        'low': row['最低'],
                        'prev_settle': row['昨结'],
                    })
            
            return unique_futures
        except Exception as e:
            err_msg = str(e)
            if 'rate limit' in err_msg.lower() or 'try again' in err_msg.lower():
                delay = base_delay * (2 ** attempt)
                print(f"  期货数据获取被限流 (attempt {attempt+1}/{max_retries})，{delay:.1f}s 后重试...")
                time.sleep(delay)
            else:
                print(f"获取国际期货数据失败: {e}")
                break
    print("获取国际期货数据最终失败")
    return []

def fetch_a_stock_list(max_retries=3, base_delay=1.0):
    """获取A股全市场列表（含行业），带重试"""
    for attempt in range(max_retries):
        try:
            import akshare as ak
            time.sleep(1.0)
            df = ak.stock_info_a_code_name()
            stocks = []
            for _, row in df.iterrows():
                code = str(row['code']).zfill(6)
                name = row['name']
                if code.startswith('60'):
                    market = 'sh'
                elif code.startswith('00'):
                    market = 'sz'
                else:
                    continue  # 跳过创业板(30)、科创板(68)、北交所等
                stocks.append({'code': code, 'name': name, 'market': market, 'industry': ''})
            return stocks
        except Exception as e:
            err_msg = str(e)
            delay = base_delay * (2 ** attempt)
            if 'rate limit' in err_msg.lower() or 'try again' in err_msg.lower():
                print(f"  A股列表获取被限流 (attempt {attempt+1}/{max_retries})，{delay:.1f}s 后重试...")
            else:
                print(f"  A股列表获取失败 (attempt {attempt+1}/{max_retries}): {e}，{delay:.1f}s 后重试...")
            time.sleep(delay)
    print("获取A股列表最终失败")
    return []

def fetch_a_stock_quotes(codes_batch, max_retries=3, base_delay=1.0):
    """从腾讯接口获取A股实时行情，带重试"""
    url = f"https://qt.gtimg.cn/q={','.join(codes_batch)}"
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            resp = urllib.request.urlopen(req, timeout=30)
            data = resp.read().decode('gbk', errors='ignore')
            return data
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  行情接口异常 (attempt {attempt+1}/{max_retries})，{delay:.1f}s 后重试...")
                time.sleep(delay)
            else:
                print(f"行情接口最终失败: {e}")
                return None
    return None

def parse_a_stock_quotes(raw_data):
    """解析A股行情数据"""
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
            }
            stocks.append(stock)
        except (ValueError, IndexError):
            continue
    return stocks

def fetch_all_a_quotes(stock_list, batch_size=80):
    """获取全市场A股实时行情，加大批次间隔"""
    all_quotes = []
    codes = [f"{s['market']}{s['code']}" for s in stock_list if s.get('market')]
    total = len(codes)
    
    for i in range(0, total, batch_size):
        batch = codes[i:i+batch_size]
        raw = fetch_a_stock_quotes(batch)
        if raw:
            quotes = parse_a_stock_quotes(raw)
            all_quotes.extend(quotes)
        if (i // batch_size + 1) % 10 == 0:
            print(f"  已获取 {min(i+batch_size, total)}/{total} 只...")
        time.sleep(0.5)  # 加大间隔，防限流
    
    return all_quotes

def fetch_a_stock_industry():
    """获取A股行业分类（申万一级）"""
    cache_path = Path('/tmp/futures_analyzer_cache')
    cache_path.mkdir(exist_ok=True)
    cache_file = cache_path / 'sw_industry_map.json'
    
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < 86400 * 7:  # 7天缓存
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    try:
        import akshare as ak
        df = ak.stock_classify_sina(symbol='申万行业')
        industry_map = {}
        for _, row in df.iterrows():
            code = str(row.get('代码', '')).zfill(6)
            industry = row.get('行业', '')
            if code and industry:
                industry_map[code] = industry
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(industry_map, f, ensure_ascii=False)
        return industry_map
    except Exception as e:
        print(f"获取行业分类失败: {e}")
        return {}

def get_stock_name_industry_map():
    """从股票名称推断行业关键词"""
    # 这个函数用于名称匹配
    industry_keywords = {
        '石油': ['石油', '石化', '油服', '海油', '能源'],
        '黄金': ['黄金', '中金', '山金', '紫金'],
        '铜': ['铜业', '铜'],
        '铝': ['铝业', '铝'],
        '锌': ['锌业', '锌'],
        '镍': ['镍'],
        '有色': ['有色', '稀土', '钨', '钼'],
        '燃气': ['燃气', '天然气'],
        '农业': ['种业', '农业', '牧业', '养殖', '饲料'],
        '食品': ['食品', '粮油', '油脂'],
        '纺织': ['纺织', '棉纺', '服饰'],
        '糖': ['糖业', '白糖'],
        '银行': ['银行'],
        '保险': ['保险'],
        '证券': ['证券', '券商'],
    }
    return industry_keywords

def match_stock_to_futures(stock_name, stock_industry, futures_keywords):
    """判断股票是否匹配期货品种"""
    name_lower = stock_name.lower()
    industry_lower = stock_industry.lower() if stock_industry else ''
    
    for keyword in futures_keywords:
        if keyword in name_lower or keyword in industry_lower:
            return True
    return False

# ============ 分析逻辑 ============

def analyze_futures(futures_list):
    """分析期货，返回涨幅最大的3个品种"""
    # 按涨跌幅排序
    futures_list.sort(key=lambda x: x['change_pct'] if x['change_pct'] is not None else -999, reverse=True)
    
    # 取涨幅最大的3个（涨跌幅>0）
    top_rising = [f for f in futures_list if f.get('change_pct', 0) > 0][:3]
    
    # 取跌幅最大的3个（用于风险提示）
    futures_list.sort(key=lambda x: x['change_pct'] if x['change_pct'] is not None else 999)
    top_falling = [f for f in futures_list if f.get('change_pct', 0) < 0][:3]
    
    return top_rising, top_falling

def analyze_a_stocks(a_quotes, industry_map, futures_mapping):
    """根据期货映射筛选A股"""
    keywords = set()
    for mapping in futures_mapping:
        keywords.update(mapping.get('keywords', []))
    
    keywords = list(keywords)
    
    matched_stocks = []
    for stock in a_quotes:
        # 只保留沪深主板
        code = stock['code']
        if not (code.startswith('60') or code.startswith('00')):
            continue
        
        industry = industry_map.get(code, '')
        
        if match_stock_to_futures(stock['name'], industry, keywords):
            # 计算评分
            score = 50
            
            # 涨幅加分（前一日强势）
            change_pct = stock.get('change_pct', 0)
            if change_pct > 5:
                score += 25
            elif change_pct > 3:
                score += 15
            elif change_pct > 1:
                score += 5
            elif change_pct < -2:
                score -= 10
            
            # 量比加分（资金关注）
            vr = stock.get('volume_ratio', 0)
            if vr > 2:
                score += 15
            elif vr > 1.5:
                score += 10
            
            # 市值适中加分（50-500亿）
            circ = stock.get('circ_market', 0)
            if 50 <= circ <= 500:
                score += 10
            
            # PE合理加分
            pe = stock.get('pe')
            if pe and 10 <= pe <= 40:
                score += 5
            
            stock['match_score'] = min(100, score)
            matched_stocks.append(stock)
    
    # 按评分排序
    matched_stocks.sort(key=lambda x: x['match_score'], reverse=True)
    return matched_stocks

def generate_report(top_rising, top_falling, matched_stocks, is_pre_market=True):
    """生成分析报告"""
    lines = []
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    lines.append(f"🌍 国际期货 → A股映射报告 - {now}")
    lines.append("=" * 70)
    lines.append("")
    
    # 期货市场概览
    lines.append("【国际期货市场 - 涨幅 TOP3】")
    if top_rising:
        for i, f in enumerate(top_rising, 1):
            lines.append(f"  {i}. {f['commodity']} ({f['code']})")
            lines.append(f"     最新价: {f['price']:.2f}  涨幅: +{f['change_pct']:.2f}%  成交量: {f['volume']}")
    else:
        lines.append("  暂无显著上涨品种")
    lines.append("")
    
    lines.append("【国际期货市场 - 跌幅 TOP3】")
    if top_falling:
        for i, f in enumerate(top_falling, 1):
            lines.append(f"  {i}. {f['commodity']} ({f['code']})")
            lines.append(f"     最新价: {f['price']:.2f}  跌幅: {f['change_pct']:.2f}%")
    else:
        lines.append("  暂无显著下跌品种")
    lines.append("")
    
    # A股映射推荐
    if is_pre_market:
        lines.append("【A股预期关注】⚠️ 当前A股尚未开盘（9:30开盘），以下为基于期货走势的预期推荐")
    else:
        lines.append("【A股实时关联】")
    lines.append("")
    
    if matched_stocks:
        lines.append(f"{'排名':<4} {'代码':<8} {'名称':<10} {'现价':<8} {'昨收涨跌':<10} {'匹配分':<8} {'流通市值(亿)':<12}")
        lines.append("-" * 70)
        for i, s in enumerate(matched_stocks[:15], 1):
            change_str = f"{s['change_pct']:+.2f}%"
            lines.append(
                f"{i:<4} {s['code']:<8} {s['name']:<10} {s['price']:<8.2f} "
                f"{change_str:<10} {s['match_score']:<8} {s.get('circ_market', 0):<12.0f}"
            )
    else:
        lines.append("  未找到匹配股票")
    
    lines.append("")
    lines.append("【筛选条件】")
    lines.append("  • 只选沪深主板（60/00开头）")
    lines.append("  • 排除创业板（30开头）和科创板（68开头）")
    lines.append("  • 匹配期货品种相关行业")
    lines.append("  • 优先前一交易日强势+资金关注度高的标的")
    lines.append("")
    lines.append("⚠️ 免责声明: 本分析仅供学习研究，不构成投资建议")
    
    return "\n".join(lines)

# ============ 主流程 ============

def run_analysis():
    """执行全部分析流程"""
    print(f"\n{'='*60}")
    print(f"国际期货 → A股映射扫描器 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # 1. 获取国际期货数据
    print("[1/4] 扫描国际期货行情...")
    futures_list = fetch_global_futures()
    if not futures_list:
        print("获取期货数据失败!")
        return None
    print(f"  获取 {len(futures_list)} 个活跃期货品种")
    
    # 2. 分析期货涨跌幅
    print("[2/4] 分析期货涨跌幅...")
    top_rising, top_falling = analyze_futures(futures_list)
    print(f"  涨幅TOP3: {', '.join([f['commodity'] for f in top_rising])}")
    print(f"  跌幅TOP3: {', '.join([f['commodity'] for f in top_falling])}")
    
    # 3. 获取A股数据
    print("[3/4] 获取A股行情...")
    a_stocks = fetch_a_stock_list()
    print(f"  沪深主板股票: {len(a_stocks)} 只")
    
    a_quotes = fetch_all_a_quotes(a_stocks)
    print(f"  成功获取 {len(a_quotes)} 只行情")
    
    # 获取行业分类
    industry_map = fetch_a_stock_industry()
    print(f"  行业分类覆盖: {len(industry_map)} 只")
    
    # 4. 映射匹配
    print("[4/4] 期货→A股映射匹配...")
    
    # 收集所有匹配的期货映射
    all_mappings = []
    for f in top_rising:
        mappings = match_futures_to_keywords(f['commodity'])
        all_mappings.extend(mappings)
    
    if all_mappings:
        matched = analyze_a_stocks(a_quotes, industry_map, all_mappings)
        print(f"  匹配到 {len(matched)} 只相关股票")
    else:
        matched = []
        print("  未找到匹配的A股映射关系")
    
    # 5. 生成报告
    now = datetime.now()
    is_pre_market = now.hour < 9 or (now.hour == 9 and now.minute < 30)
    report = generate_report(top_rising, top_falling, matched, is_pre_market)
    
    return report

# ============ 入口 ============

if __name__ == '__main__':
    report = run_analysis()
    if report:
        print("\n" + report)
        # 保存报告
        report_path = f"/tmp/futures_a_share_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n报告已保存: {report_path}")
        
        # 发送邮件通知
        try:
            skill_dir = '/root/.openclaw/workspace/skills/imap-smtp-email-chinese'
            result = subprocess.run(
                [
                    'node', f'{skill_dir}/scripts/smtp.js', 'send',
                    '--to', 'hansonnan.ding@sc.com,leo.sun@sc.com,jacky.an@sc.com',
                    '--subject', f'国际期货→A股映射报告 - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
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
