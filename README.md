# ToB科技客户智能分级与销售情报系统

> 基于 A股990家科技上市公司公开数据，构建"**数据筛选 + AI尽调**"两层架构的 ToB 销售智能系统：
> 第一层用爬虫+评分模型从全市场筛出高价值客户，第二层用 AI Agent 联网搜索做单客户深度尽调。

## 一、项目背景

ToB 销售的两大核心问题：**找客户**（市场上谁值得跟）和**懂客户**（锁定后怎么打）。
本项目模拟真实销售工作流，用数据工程解决"找客户"的效率问题，用 AI 解决"懂客户"的深度问题。

## 二、系统架构（两层漏斗）

```
全市场 990 家科技公司（巨潮/同花顺/腾讯财经 多源爬虫）
        ↓ 12维度 AHP 层次分析法评分 + 6类需求信号识别
TOP 高价值客户（A/B/C/D 四级分级）
        ↓ Tavily 联网搜索（新闻/高管/融资/采购 5维度）
        ↓ DeepSeek 大模型综合分析
单客户深度销售情报（需求解读/决策人/方案/话术/邮件）
```

## 三、核心功能

- **多源数据采集**：requests + BeautifulSoup 多线程爬虫，覆盖巨潮资讯网（证监会指定披露平台）、同花顺F10、腾讯财经，990家公司、市值数据覆盖率99.7%
- **12维度评分模型**：AHP层次分析法确定权重，从规模/增长/盈利/财务健康/行业景气等12个维度打分，A/B/C/D四级分级
- **需求信号识别**：从财务和业务数据中识别扩张型/降本型/转型型/合规型/技术升级型/建设型6类需求，量化需求强度
- **可视化分析看板**：Streamlit + Plotly，数据总览、行业深度对比、评分可解释性（维度贡献度/vs行业平均）、交互式筛选
- **AI快速情报**：基于结构化数据10秒生成销售情报
- **AI联网深度尽调**：自动生成搜索query、Tavily实时联网检索、大模型综合成带信息来源的深度尽调报告（模拟销售代表Agent的思考链）

## 四、技术栈

| 层 | 技术 |
|----|------|
| 数据采集 | Python, Requests, BeautifulSoup4, 多线程并发 |
| 数据处理 | Pandas, NumPy, 正则清洗 |
| 评分模型 | AHP层次分析法, 特征工程, 加权评分 |
| AI应用 | DeepSeek LLM, Tavily Search API, Prompt Engineering |
| 可视化 | Streamlit, Plotly |

## 五、项目结构

```
customer_scoring/
├── src/
│   ├── data_collector.py        # 公司基本信息爬虫（巨潮+同花顺）
│   ├── financial_collector.py   # 财务数据爬虫
│   ├── extra_data_collector.py  # 市值数据采集（腾讯财经）
│   ├── scoring_model.py         # 12维度AHP评分模型
│   ├── demand_analyzer.py       # 需求信号识别
│   └── company_researcher.py    # AI联网深度尽调（Tavily+DeepSeek）
├── data/                        # 原始/中间数据
├── outputs/                     # 评分结果、需求分析结果
├── research_reports/            # AI尽调报告
├── app.py                       # Streamlit可视化看板
├── requirements.txt
└── README.md
```

## 六、快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 数据采集（可跳过，outputs里已有结果）
python src/data_collector.py
python src/financial_collector.py
python src/extra_data_collector.py

# 3. 评分与需求分析
python src/scoring_model.py
python src/demand_analyzer.py

# 4. 启动看板
streamlit run app.py
```

## 七、数据来源与免责声明

- 巨潮资讯网（证监会指定上市公司信息披露平台）、同花顺F10、腾讯财经、Tavily联网搜索
- 所有数据均来自公开渠道，AI分析基于公开信息推断，仅供学习研究，不构成投资或销售决策依据
