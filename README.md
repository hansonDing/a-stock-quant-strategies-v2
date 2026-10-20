# A股量化策略合集（真实数据版）

基于 Python 的 A 股量化交易策略，使用 **腾讯财经真实历史数据** 进行模型训练和回测。

> ⚠️ 与原版最大的区别：所有策略文件现在使用真实 K 线数据训练模型，不再使用模拟随机数据做"假预测"。

---

## 📁 项目结构

```
a-stock-quant-strategies/
├── strategies/                    # 策略代码
│   ├── trend_following/
│   │   └── trend_backtest.py      # 双均线/动量/MACD（真实数据回测）
│   └── multi_factor/
│       ├── ml_factor_tencent_3year.py     # LightGBM + 3年真实数据（推荐）⭐
│       ├── ml_factor_tencent_1year.py     # LightGBM + 1年真实数据（快速验证）
│       ├── tomorrow_picks.py              # 明日选股推荐（需已训练模型）
│       ├── ml_factor_backtest.py          # ⚠️ 模拟数据演示（仅供学习）
│       └── ml_factor_hybrid.py            # ⚠️ 模拟训练+实时预测（仅供学习）
├── projects/                      # 盘前扫描器（定时任务用）
│   ├── a_stock_analyzer/          # A股强势扫描器（剔除创业板/科创板）
│   ├── global_futures_analyzer/   # 国际期货→A股映射
│   ├── tech_sector_scanner/       # 科技板块晨间扫描
│   └── spring-ai-rag/             # submodule → hansonDing/spring-ai-rag
├── utils/                         # 公共工具模块
│   ├── api.py                     # 腾讯实时行情 + 历史K线（统一封装）
│   ├── factors.py                 # 多因子计算（18个因子统一实现）
│   ├── backtest.py                # 通用回测引擎（含交易成本）
│   ├── tencent_realtime_quote.py  # 兼容层 → api.py
│   ├── tencent_history_data.py    # 兼容层 → api.py
│   └── analyze_today.py           # 今日收盘数据分析
├── daily_stock_picks.py           # 每日选股推荐（真实模型）⭐
├── demo.py                        # NL2SQL RAG 演示（依赖见 requirements-rag.txt）
├── a-stock-quant-v3/              # submodule → hansonDing/a-stock-quant-v3
├── data/                          # 数据输出目录（gitignore，含模型文件）
├── reports/                       # 回测报告图表
├── requirements.txt               # 量化依赖
├── requirements-rag.txt           # NL2SQL 演示的 RAG 依赖（可选）
├── .gitignore
└── README.md
```

> clone 后需拉取子模块：`git submodule update --init`
> 注意：`data/` 不入 git（含训练好的模型），全新环境首次运行自动重新训练

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 依赖包见 `requirements.txt`

### 安装依赖

```bash
# 量化策略（核心）
pip install -r requirements.txt

# NL2SQL 演示（可选，依赖较重含 torch）
pip install -r requirements-rag.txt
```

> 建议使用虚拟环境：`python -m venv venv && source venv/bin/activate`

### 运行策略

```bash
# 1. 趋势跟踪策略回测（真实数据）
python strategies/trend_following/trend_backtest.py

# 2. 多因子 + LightGBM 回测（1年真实数据，快速验证）
python strategies/multi_factor/ml_factor_tencent_1year.py

# 3. 多因子 + LightGBM 回测（3年真实数据 + Walk-Forward，推荐）
python strategies/multi_factor/ml_factor_tencent_3year.py

# 4. 每日选股推荐（首次运行会自动训练模型）
python daily_stock_picks.py

# 5. 明日选股推荐（复用已训练模型）
python strategies/multi_factor/tomorrow_picks.py

# 6. 获取今日收盘数据分析
python utils/analyze_today.py
```

---

## 📊 策略说明

### 一、趋势跟踪策略

| 策略 | 原理 | 适用场景 |
|------|------|----------|
| **双均线交叉 (MA5/MA20)** | 短期均线上穿长期均线买入，下穿卖出 | 中期趋势 |
| **动量策略 (20日)** | 过去20日收益率为正则持有 | 趋势明显的市场 |
| **MACD策略 (12/26/9)** | DIF上穿DEA买入，下穿卖出 | 通用趋势判断 |

> 数据源：腾讯财经真实 K 线（前复权）

### 二、多因子 + LightGBM 策略（推荐）

#### 因子体系（18个因子）

| 因子类别 | 因子 | 说明 |
|----------|------|------|
| **动量因子** | mom_5, mom_10, mom_20, mom_60, mom_accel | 多周期价格动量 |
| **波动率因子** | vol_5, vol_10, vol_20, vol_ratio | 多周期波动率及比率 |
| **技术因子** | rsi_14, macd_dif, macd_dea, macd_hist | RSI、MACD指标 |
| **均线偏离** | bias_5, bias_10, bias_20, bias_60 | 多周期均线偏离度 |
| **流动性因子** | turnover_ma5, amihud | 换手率、非流动性 |

#### 回测方法

- **1年版**：前 60% 训练，后 40% 测试
- **3年版**：Walk-Forward 滚动回测（每 60 天重训一次模型），更接近实盘

#### 交易成本

回测已纳入双边交易成本 0.3%（佣金 + 印花税 + 滑点），更接近真实收益。

---

## 🔌 数据接口

### 腾讯实时行情接口

```
接口地址: https://qt.gtimg.cn/q=股票代码
支持市场: A股(沪/深)、港股、美股
返回格式: GBK编码，分号分隔
```

**股票代码格式：**
- 沪市: `sh600519`（贵州茅台）
- 深市: `sz000858`（五粮液）
- 港股: `hk00700`（腾讯控股）
- 美股: `aapl`（苹果）

### 腾讯历史K线接口

```
接口地址: https://web.ifzq.gtimg.cn/appstock/app/fqkline/get
参数格式: ?param=代码,周期,开始日期,结束日期,复权类型,qfq
```

**参数说明：**
- 周期: `day`(日线)、`week`(周线)、`month`(月线)
- 复权: `640`=前复权、`0`=不复权
- 返回字段: 日期、开盘、收盘、最低、最高、成交量、成交额

---

## 🛠 主要改进（对比原版）

| 问题 | 原版 | 当前版本 |
|------|------|----------|
| 模型训练 | 用随机模拟数据 | ✅ 用腾讯真实 3 年 K 线 |
| 实时预测因子 | 用今日涨跌幅"冒充"历史因子 | ✅ 拉取 120 天历史计算真实因子 |
| 回测数据 | 模拟生成 | ✅ 腾讯真实 K 线 |
| 代码重复 | 5 处重复实现 | ✅ 统一抽成 `utils/api.py` / `factors.py` |
| 绝对路径 | `/data/user/work` 硬编码 | ✅ 使用 `PROJECT_ROOT / 'data'` 相对路径 |
| 依赖管理 | 无版本锁定 | ✅ `requirements.txt` 锁定版本 |
| 异常处理 | `try: ... except: return None` 吞异常 | ✅ 分级日志 + 重试机制 |
| 交易成本 | 未考虑 | ✅ 双边 0.3% 成本纳入 |
| 回测方法 | 固定训练/测试切分 | ✅ Walk-Forward 滚动回测 |

---

## ⏰ 每日定时任务流水线

项目通过 **OpenClaw cron** 调度，在工作日盘前自动执行扫描与预测。

### 执行机制

不是传统 crontab 直接跑脚本，而是 OpenClaw 的 agent 调度：

```
到点触发（cron 表达式，Asia/Shanghai 时区）
  → 创建隔离会话（isolated session）
  → AI agent 接收任务指令，执行对应 python 脚本
  → 结果推送到 Kimi 聊天（部分任务同步发邮件）
```

任务定义文件：`~/.openclaw/cron/jobs.json`；运行日志：`~/.openclaw/cron/runs/<任务ID>.jsonl`

### 任务一览（工作日 周一~五）

| 时间 | 任务 | 脚本 | 结果推送 |
|------|------|------|----------|
| 08:17 | 国际期货→A股映射扫描器 | `projects/global_futures_analyzer/futures_a_share_mapper.py` | Kimi + 邮件 |
| 08:25 | 科技板块晨间扫描器 | `projects/tech_sector_scanner/tech_sector_scanner.py` | Kimi + 邮件 |
| 08:30 | A股强势扫描器（剔除创业板/科创板） | `projects/a_stock_analyzer/a_stock_scanner.py` | Kimi + 邮件 |
| 09:07 | 科技股晨间预测推送（LightGBM） | `strategies/multi_factor/tomorrow_picks.py` | Kimi |

邮件接收人：`hansonnan.ding@sc.com`、`leo.sun@sc.com`、`jacky.an@sc.com`

### 手动运行

以下命令均在**仓库根目录**执行：

```bash
# 国际期货→A股映射扫描
python3 projects/global_futures_analyzer/futures_a_share_mapper.py

# 科技板块扫描
python3 projects/tech_sector_scanner/tech_sector_scanner.py

# A股强势扫描（剔除创业板/科创板）
python3 projects/a_stock_analyzer/a_stock_scanner.py

# LightGBM 明日预测（前置条件：data/lgb_model.pkl 已存在）
# 全新环境需先训练：python daily_stock_picks.py（会自动训练并保存模型）
python strategies/multi_factor/tomorrow_picks.py
```

> 扫描器脚本输出报告文本到 stdout；定时任务由 agent 抓取结果推送（Kimi 聊天 + 邮件），手动运行时脚本本身不发邮件。
> 依赖：扫描器仅需 akshare + pandas + requests（见 requirements.txt），明日预测额外需要 lightgbm + scikit-learn + joblib。

### 任务管理

```bash
openclaw cron list            # 查看所有任务及下次运行时间
openclaw cron list --verbose  # 查看任务完整定义
```

> 另一个仓库 [a-stock-quant-v3](https://github.com/hansonDing/a-stock-quant-v3) 承担 07:50 的多因子选股推荐和周日的缓存清理，通过 submodule 关联。

---

## ⚠️ 免责声明

本项目仅供学习研究参考，不构成任何投资建议。量化策略基于历史数据回测，过往业绩不代表未来表现。实盘交易存在交易成本、滑点、涨跌停限制等回测未完全考虑的因素，实际收益可能显著低于回测结果。投资有风险，入市需谨慎。

---

## 📄 License

MIT License
