# -*- coding: utf-8 -*-
"""
AI 联网深度尽调模块（企业客户 12 维搜索 → 9 段销售尽调报告）
流程：生成维度 query → Tavily advanced 搜索（URL 去重）→ DeepSeek 综合报告
运行：python src/company_researcher.py
"""
import requests
import json
import time
import os
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")


def _get_tavily_key():
    return os.getenv("TAVILY_API_KEY", "") or TAVILY_API_KEY


def _get_deepseek_key():
    return os.getenv("DEEPSEEK_API_KEY", "") or DEEPSEEK_API_KEY

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_MODULE_DIR)  # 项目根目录
REPORT_DIR = os.path.join(BASE_DIR, 'research_reports')
try:
    os.makedirs(REPORT_DIR, exist_ok=True)
except Exception:
    pass

# 近 6 个月（约 180 天）——维度 2、6 使用
RECENT_DAYS = 180

# 企业客户尽调：12 个搜索维度
# recent_only=True → 限定近 6 个月，避免旧闻
SEARCH_DIMENSIONS = [
    {
        'id': 1,
        'name': '公司背景与主营业务',
        'recent_only': False,
        'query_tpl': '{company} 公司简介 主营业务 核心产品 市场地位 发展历程',
    },
    {
        'id': 2,
        'name': '最新动态与新闻（近6个月）',
        'recent_only': True,
        'query_tpl': '{company} 最新动态 新闻 业务进展 战略 近半年',
    },
    {
        'id': 3,
        'name': '组织架构与决策链',
        'recent_only': False,
        'query_tpl': '{company} 组织架构 管理部门 决策流程 信息化分管领导',
    },
    {
        'id': 4,
        'name': '关键决策人',
        'recent_only': False,
        'query_tpl': '{company} 董事长 总经理 CEO CIO 信息中心主任 高管团队',
    },
    {
        'id': 5,
        'name': '现有IT系统与供应商',
        'recent_only': False,
        'query_tpl': '{company} 信息化 现有系统 ERP OA 超融合 云 供应商 中标',
    },
    {
        'id': 6,
        'name': '采购与预算信号（近6个月）',
        'recent_only': True,
        'query_tpl': '{company} 招标 采购 中标 预算 IT项目 信息化建设 近半年',
    },
    {
        'id': 7,
        'name': '行业政策与痛点',
        'recent_only': False,
        'query_tpl': '{industry} 行业 政策 数字化 痛点 合规 信创 {company}',
    },
    {
        'id': 8,
        'name': '财务与经营状况',
        'recent_only': False,
        'query_tpl': '{company} 财报 营收 利润 经营状况 业绩 资产负债',
    },
    {
        'id': 9,
        'name': '竞品与市场竞争格局',
        'recent_only': False,
        'query_tpl': '{company} 竞争对手 竞品 市场份额 替代 选型对比 {product_kw}',
    },
    {
        'id': 10,
        'name': '融资与资本动态',
        'recent_only': False,
        'query_tpl': '{company} 融资 投资 股权 上市 定增 资本运作',
    },
    {
        'id': 11,
        'name': '技术投入与研发动态',
        'recent_only': False,
        'query_tpl': '{company} 研发 技术投入 专利 数字化转型 AI 创新',
    },
    {
        'id': 12,
        'name': '客户与上下游生态',
        'recent_only': False,
        'query_tpl': '{company} 客户案例 合作伙伴 供应链 上下游 生态联盟',
    },
]


def extract_product_keywords(our_product):
    """从产品描述提取搜索关键词。"""
    if not our_product:
        return ''
    text = str(our_product)
    for prefix in ['核心能力：', '产品亮点：', '适配行业：', '目标客户规模：', '产品类别：']:
        if prefix in text:
            text = text[:text.index(prefix)]
    for word in ['解决方案', '系统', '平台', '软件', '服务', '产品', '套件', '方案', '与', '和', '及']:
        text = text.replace(word, ' ')
    for punct in ['（', '）', '(', ')', '。', '，', ',', '：', ':', '；', ';', '、', '/', '\\', '-', '—']:
        text = text.replace(punct, ' ')
    keywords = ' '.join(text.split())
    return keywords[:40].strip()


def tavily_search(query, max_results=5, recent_only=False):
    """
    Tavily advanced 搜索。
    - search_depth=advanced
    - 每条内容截取 1500 字
    - recent_only 时限定近 180 天
    """
    if not _get_tavily_key():
        print("  ⚠️ 未配置 TAVILY_API_KEY")
        return []

    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {_get_tavily_key()}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    if recent_only:
        # 近 6 个月：优先用 days；部分账号也支持 time_range
        payload["days"] = RECENT_DAYS
        payload["topic"] = "news"

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get('results', []):
            content = item.get('content') or item.get('raw_content') or ''
            content = str(content)[:1500]
            results.append({
                'title': item.get('title', '') or '',
                'url': item.get('url', '') or '',
                'content': content,
                'published_date': item.get('published_date') or item.get('published_date', '') or '',
            })
        return results
    except Exception as e:
        # days 参数不被支持时，降级重试（仍靠 query 中的「近半年」约束）
        should_retry = recent_only and (
            'days' in str(e).lower()
            or (getattr(e, 'response', None) is not None and getattr(e.response, 'status_code', None) == 400)
        )
        if should_retry:
            try:
                payload.pop('days', None)
                payload.pop('topic', None)
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                results = []
                for item in data.get('results', []):
                    content = item.get('content') or ''
                    results.append({
                        'title': item.get('title', '') or '',
                        'url': item.get('url', '') or '',
                        'content': str(content)[:1500],
                        'published_date': item.get('published_date') or '',
                    })
                return results
            except Exception as e2:
                print(f"  ⚠️ 搜索失败 [{query}]：{e2}")
                return []
        print(f"  ⚠️ 搜索失败 [{query}]：{e}")
        return []


def build_dimension_queries(company_name, industry='', our_product='', low_score_dims=None):
    """生成 12 维 query；低分维度追加一条加深度搜索。"""
    product_kw = extract_product_keywords(our_product) or '信息化 数字化'
    industry = industry or '企业'
    queries = []
    for dim in SEARCH_DIMENSIONS:
        q = dim['query_tpl'].format(
            company=company_name,
            industry=industry,
            product_kw=product_kw,
        )
        queries.append({
            'dim_id': dim['id'],
            'dim_name': dim['name'],
            'recent_only': dim['recent_only'],
            'query': q,
            'boost': False,
        })

    # 补充：围绕评分低分维度加搜（若有）
    if low_score_dims:
        for dim_label in low_score_dims[:3]:
            queries.append({
                'dim_id': 0,
                'dim_name': f'低分维度深挖-{dim_label}',
                'recent_only': False,
                'query': f"{company_name} {dim_label} 现状 问题 改进 采购 机会",
                'boost': True,
            })
    return queries


def research_company(
    company_name,
    industry='',
    our_product='',
    verbose=True,
    low_score_dims=None,
    progress_callback=None,
):
    """
    12 维联网尽调。按 URL 全局去重，返回情报列表。
    progress_callback(done, total, message) 可选，供 Streamlit 进度条。
    """
    if verbose:
        print(f"\n🔍 开始联网尽调：{company_name}")
        if our_product:
            print(f"   我方产品：{our_product}")
        print("-" * 50)

    query_specs = build_dimension_queries(company_name, industry, our_product, low_score_dims)
    total = len(query_specs)
    all_intel = []
    seen_urls = set()

    for i, spec in enumerate(query_specs, 1):
        if progress_callback:
            progress_callback(i - 1, total, f"搜索：{spec['dim_name']}")
        if verbose:
            tag = "近6月" if spec['recent_only'] else "全量"
            print(f"  [{i}/{total}] ({tag}) {spec['dim_name']}：{spec['query']}")

        results = tavily_search(
            spec['query'],
            max_results=5 if spec.get('boost') else 4,
            recent_only=spec['recent_only'],
        )
        added = 0
        for r in results:
            url = (r.get('url') or '').strip()
            # URL 去重（无 URL 时用标题+内容前 80 字兜底）
            dedup_key = url.lower() if url else f"{r.get('title','')}|{r.get('content','')[:80]}"
            if not dedup_key or dedup_key in seen_urls:
                continue
            seen_urls.add(dedup_key)
            all_intel.append({
                '搜索维度': spec['dim_name'],
                '维度编号': spec['dim_id'],
                '标题': r.get('title', ''),
                '来源': url,
                '内容': r.get('content', ''),
                '发布日期': r.get('published_date', ''),
                '查询语句': spec['query'],
            })
            added += 1

        if verbose:
            print(f"        新增 {added} 条（去重后累计 {len(all_intel)}）")
        time.sleep(0.35)

    if progress_callback:
        progress_callback(total, total, f"搜索完成，共 {len(all_intel)} 条")
    if verbose:
        print(f"\n✅ 共收集 {len(all_intel)} 条去重情报")
    return all_intel


def _format_score_overview(score_overview):
    """把评分概览格式化为 prompt 文本。"""
    if not score_overview:
        return "（未提供客户评分数据）"
    lines = []
    total = score_overview.get('总分', '')
    grade = score_overview.get('客户分级', '')
    lines.append(f"- 总分：{total}　分级：{grade}")
    dims = score_overview.get('维度得分') or []
    if dims:
        lines.append("- 各维度得分：")
        for d in dims:
            name = d.get('维度', '')
            sc = d.get('得分', '')
            w = d.get('权重', '')
            flag = d.get('是否低分', False)
            mark = " ← 低分，需深挖" if flag else ""
            lines.append(f"  · {name}：{sc}分（权重 {w}）{mark}")
    low = score_overview.get('低分维度') or []
    if low:
        lines.append("- 本次尽调重点低分维度：" + "、".join(low))
    return "\n".join(lines)


def _format_sales_followup(sales_followup):
    """
    单独格式化销售跟进一手情报。
    明确标注优先级高于联网搜索结果。
    """
    if not sales_followup:
        return (
            "【销售内部一手情报】\n"
            "（本客户暂无销售跟进记录）\n"
            "说明：若后续有跟进表数据，其优先级高于联网搜索结果。"
        )

    # 支持 dict 或 list[dict]
    rows = sales_followup if isinstance(sales_followup, list) else [sales_followup]
    field_order = [
        '客户名称', '合作成熟度阶段', '现有供应商名称', '绑定程度', '竞争激烈程度',
        '客户抱怨现有产品', '是否行业标杆', '行业排名', '跟进日期', '跟进人', '备注',
    ]

    lines = [
        "【销售内部一手情报】",
        "⚠️ 以下为销售内部一手情报，优先级高于联网搜索结果。",
        "当与联网公开信息冲突时：以本段销售情报为准，并在报告中标注「销售一手情报」；",
        "联网信息仅作补充或交叉验证。",
        "",
    ]
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            continue
        if len(rows) > 1:
            lines.append(f"—— 跟进记录 #{i} ——")
        has_any = False
        for key in field_order:
            if key not in row:
                continue
            val = row.get(key)
            if val is None or str(val).strip() in ('', 'nan', 'None', 'NaN'):
                continue
            lines.append(f"- {key}：{val}")
            has_any = True
        # 其他未列字段也带上
        for key, val in row.items():
            if key in field_order:
                continue
            if val is None or str(val).strip() in ('', 'nan', 'None', 'NaN'):
                continue
            lines.append(f"- {key}：{val}")
            has_any = True
        if not has_any:
            lines.append("- （记录存在但字段均为空）")
        lines.append("")
    return "\n".join(lines).rstrip()


def generate_deep_report(
    company_name,
    intel_list,
    our_product,
    company_basic=None,
    score_overview=None,
    sales_followup=None,
):
    """
    基于联网情报 + 销售跟进一手情报生成 9 段深度尽调报告。
    销售跟进数据单独成块，优先级高于联网搜索。
    """
    # 按维度分组整理情报
    by_dim = {}
    for item in intel_list:
        dim = item.get('搜索维度') or '未分类'
        by_dim.setdefault(dim, []).append(item)

    intel_text = ""
    idx = 1
    for dim, items in by_dim.items():
        intel_text += f"\n### 维度：{dim}（{len(items)}条）\n"
        for item in items:
            intel_text += f"\n【情报{idx}】\n"
            intel_text += f"标题：{item.get('标题', '')}\n"
            intel_text += f"来源：{item.get('来源', '')}\n"
            if item.get('发布日期'):
                intel_text += f"日期：{item.get('发布日期')}\n"
            intel_text += f"内容：{item.get('内容', '')}\n"
            idx += 1

    basic_text = ""
    if company_basic:
        basic_text = f"""
【公司结构化数据（评分系统/公开数据，不含销售跟进字段）】
{json.dumps(company_basic, ensure_ascii=False, indent=2)}
"""

    score_text = _format_score_overview(score_overview)
    sales_text = _format_sales_followup(sales_followup)
    today = datetime.now().strftime('%Y-%m-%d')
    six_months_ago = (datetime.now() - timedelta(days=RECENT_DAYS)).strftime('%Y-%m-%d')

    prompt = f"""# 角色
你是资深 ToB 销售总监兼行业分析师，擅长把「销售一手情报 + 联网公开情报」转成可执行的客户尽调与销售行动方案。

# 任务
为「{company_name}」撰写深度客户尽调报告。
我方产品/方案：{our_product}
报告日期：{today}
时间敏感的公开信息请优先采用 {six_months_ago} 至今的内容；更早信息仅作背景，并标注可能过时。

# 客户评分概览（来自内部评分模型，必须在报告开头呈现并围绕低分维度深挖）
{score_text}

{basic_text}

# ========== 销售内部一手情报（最高优先级）==========
{sales_text}
# ========== 以上销售情报结束 ==========

# 联网搜索到的公开情报（已按维度分组、URL 去重；优先级低于销售一手情报）
{intel_text if intel_text.strip() else "（本次未检索到联网情报）"}

# 硬性要求
1. **情报优先级**：销售内部一手情报 > 结构化评分数据 > 联网搜索结果。
   冲突时以销售一手情报为准，并写明「据销售一手情报…」；联网信息用于补充或交叉验证。
2. **销售跟进字段必须单独使用**：现有供应商、绑定程度、竞争激烈程度、合作成熟度、客户抱怨、行业标杆/排名、跟进备注等，优先写入第四、七、八部分，不要淹没在公开信息里。
3. **只基于上方已给信息**写结论；公开情报中的事实尽量标注来源 URL 或标题。
4. 某方面情报不足时，必须明确写：**「公开情报有限，需进一步调研」**；销售字段为空时写「销售侧尚未登记」，禁止编造供应商名、合同到期日、决策人联系方式等。
5. 区分【事实】与【推断】；推断需说明依据。
6. 尽调重点围绕**低分维度**展开，解释可能原因与销售可验证动作。
7. **第四部分（IT 现状与现有供应商）**和**第七部分（竞争态势与替换机会）**是销售最关心的内容：
   - 必须尽量写具体：现有供应商是谁、系统类型、绑定/竞争态势、替换窗口（有则写，无则标明需调研）
   - 写清从哪切入、替换动机、对手弱项
8. **第八部分**必须给出可执行的 **30/60/90 天行动计划** + **破冰话术**（优先引用销售备注与真实动态），禁止空话套话。
9. 严格按下列结构输出（可用二级标题），不要增减大标题：

## 〇、客户评分概览与尽调重点
（总分/分级；低分维度列表；本次尽调要验证的 3 个关键问题）

## 〇-附、销售一手情报摘要
（单独复述销售跟进关键字段；若无记录则写「暂无销售跟进登记」）

## 一、公司概况与行业地位
## 二、最新动态与业务风向
（优先近 6 个月；逐条列动态并标注时间/来源；说明采购信号）
## 三、组织架构与关键决策人
（决策链；关键人职位与职责推断；接触路径建议；信息不足处标明）
## 四、IT 现状与现有供应商
（优先采用销售登记的供应商/绑定/抱怨；再补充联网线索；合同或项目时间线；与我方产品的替换/互补关系）
## 五、财务与经营分析
（结合结构化数据与搜索情报；财务健康与预算能力判断）
## 六、需求与痛点分析
（痛点、紧迫性、与我方产品匹配表：客户需求 / 产品能力 / 匹配度 / 切入点）
## 七、竞争态势与替换机会
（优先销售登记的竞争激烈程度/绑定程度；竞品格局；替换窗口；切入角度与风险）
## 八、销售策略与行动建议
必须包含：
- 30 天行动计划（具体动作、目标对象、产出物）
- 60 天行动计划
- 90 天行动计划
- 破冰话术（至少 2 套，优先引用销售备注与真实动态）
- 首封邮件/微信草稿（含主题）
- 风险与合规注意点
## 九、情报来源清单
（分两类列出：①销售一手情报字段；②联网 URL；并注明各维度情报充分度：充分 / 一般 / 不足）

---
⚠️ 本报告基于销售内部登记与公开联网信息整理，仅供内部销售参考；决策人个人信息以官方披露为准，禁止用于不当用途。"""

    if not _get_deepseek_key():
        return "❌ 未配置 DEEPSEEK_API_KEY，无法生成报告。"

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {_get_deepseek_key()}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.45,
        "max_tokens": 8000,
    }
    resp = requests.post(url, headers=headers, json=data, timeout=180)
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content']


def save_research_outputs(company_name, our_product, report, intel_list):
    """保存报告与原始情报到 research_reports/。"""
    safe = "".join(c for c in company_name if c not in '\\/:*?"<>|')
    report_file = os.path.join(REPORT_DIR, f"{safe}_深度尽调报告.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# {company_name} 深度尽调报告\n\n")
        f.write(f"我方产品：{our_product}\n\n")
        f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(report)
    intel_file = os.path.join(REPORT_DIR, f"{safe}_原始情报.json")
    with open(intel_file, 'w', encoding='utf-8') as f:
        json.dump(intel_list, f, ensure_ascii=False, indent=2)
    return report_file, intel_file


def main():
    company_name = "达梦数据"
    our_product = "超融合基础设施 / 企业级云与安全方案"
    industry = "软件"

    intel = research_company(
        company_name,
        industry=industry,
        our_product=our_product,
        low_score_dims=['合作成熟度', '显性需求信号'],
    )
    if not intel:
        print("❌ 没有搜索到情报，请检查 Tavily API Key / 网络")
        return

    print(f"\n🤖 DeepSeek 综合 {len(intel)} 条情报生成报告...")
    overview = {
        '总分': 58,
        '客户分级': 'B级（重点跟进）',
        '低分维度': ['合作成熟度', '显性需求信号'],
        '维度得分': [
            {'维度': '合作成熟度', '得分': 25, '权重': '10.7%', '是否低分': True},
            {'维度': '显性需求信号', '得分': 20, '权重': '12.0%', '是否低分': True},
        ],
    }
    report = generate_deep_report(
        company_name, intel, our_product,
        company_basic={'所属行业': industry, '总分': 58},
        score_overview=overview,
    )
    rf, iff = save_research_outputs(company_name, our_product, report, intel)
    print("=" * 60)
    print("✅ 尽调完成")
    print(f"  报告：{rf}")
    print(f"  情报：{iff}")
    print("=" * 60)
    print(report[:2000])


if __name__ == '__main__':
    main()
