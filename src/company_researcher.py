"""
AI联网深度尽调模块（模拟销售代表Agent）
流程：选定公司 → 自动生成搜索query → Tavily联网搜索 → DeepSeek综合分析 → 深度尽调报告
运行：python src/company_researcher.py
"""
import requests
import json
import time
import os

# ========== API Key配置 ==========
# 从环境变量读取（本地由 .env 提供，云端由 app.py 同步 st.secrets）
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")


# 输出目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(BASE_DIR, 'research_reports')
os.makedirs(REPORT_DIR, exist_ok=True)


def tavily_search(query, max_results=5):
    """调用Tavily联网搜索，返回结果列表"""
    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get('results', []):
            results.append({
                'title': item.get('title', ''),
                'url': item.get('url', ''),
                'content': item.get('content', '')[:800]  # 每条摘要最多800字
            })
        return results
    except Exception as e:
        print(f"  ⚠️ 搜索失败 [{query}]：{e}")
        return []


def generate_search_queries(company_name, industry=''):
    """根据公司名自动生成5个维度的搜索query（模拟销售代表的思考链）"""
    queries = [
        f"{company_name} 公司背景 主营业务 市场地位 核心产品",
        f"{company_name} 最新消息 新闻 动态 2026",
        f"{company_name} 财报 业绩 营收 融资 增长",
        f"{company_name} 董事长 CEO 高管 创始人 管理团队",
        f"{company_name} 数字化转型 采购 招标 IT建设 合作"
    ]
    return queries


def research_company(company_name, industry='', verbose=True):
    """
    对一家公司做联网尽调，返回搜索到的原始情报
    """
    if verbose:
        print(f"\n🔍 开始联网尽调：{company_name}")
        print("-" * 50)
    
    queries = generate_search_queries(company_name, industry)
    all_intel = []
    
    for i, q in enumerate(queries, 1):
        if verbose:
            print(f"  [{i}/5] 搜索：{q}")
        results = tavily_search(q, max_results=4)
        if verbose:
            print(f"        获取 {len(results)} 条结果")
        
        for r in results:
            all_intel.append({
                '搜索维度': q.split(company_name)[-1].strip(),
                '标题': r['title'],
                '来源': r['url'],
                '内容': r['content']
            })
        time.sleep(0.5)  # 礼貌延时
    
    if verbose:
        print(f"\n✅ 共收集 {len(all_intel)} 条情报")
    
    return all_intel


def generate_deep_report(company_name, intel_list, our_product, company_basic=None):
    """把搜索情报喂给DeepSeek，生成深度尽调报告"""
    
    # 整理情报文本
    intel_text = ""
    for i, item in enumerate(intel_list, 1):
        intel_text += f"\n【情报{i}】维度：{item['搜索维度']}\n"
        intel_text += f"标题：{item['标题']}\n"
        intel_text += f"来源：{item['来源']}\n"
        intel_text += f"内容：{item['内容']}\n"
    
    basic_text = ""
    if company_basic:
        basic_text = f"""
【公司结构化数据（来自公开财报）】
{json.dumps(company_basic, ensure_ascii=False, indent=2)}
"""
    
    prompt = f"""# 角色
你是资深ToB销售总监，擅长基于联网情报对目标客户做深度尽调。

# 任务
以下是通过联网搜索收集到的关于「{company_name}」的真实情报，以及结构化财务数据。
我方产品是：{our_product}
请基于这些**真实搜索到的信息**，生成一份深度客户尽调报告。

{basic_text}

# 联网搜索到的真实情报
{intel_text}

# 要求
1. 必须基于上方搜索情报分析，情报里提到的事实要标注来源
2. 如果某个方面搜索情报不足，明确写"公开情报有限，需进一步调研"，不要编造
3. 严格按以下结构输出：

## 一、公司概况（基于搜索情报）
- 主营业务和核心产品：
- 市场地位和竞争对手：
- 发展历程和关键里程碑：

## 二、最新动态与业务风向（重点，基于新闻情报）
- 近期重要动态（逐条列出，标注时间/来源）：
- 战略方向和业务重心变化：
- 这些动态背后释放的采购信号：

## 三、关键决策人（基于高管情报）
- 创始人/董事长/CEO背景：
- 技术/业务负责人：
- 决策风格推断：

## 四、需求与痛点分析（核心）
- 基于动态和财报推断的真实需求：
- 需求紧迫度和触发事件：
- 与我方产品（{our_product}）的匹配点：

## 五、销售策略建议
- 最佳切入角度（结合最新动态找钩子）：
- 破冰话术（引用对方近期动态，让对方觉得你做过功课）：
- 风险点和注意事项：

## 六、情报来源清单
（列出本次分析引用的主要信息来源URL）

---
⚠️ 本报告基于公开联网信息整理，决策人个人信息以官方披露为准。"""

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "max_tokens": 4500
    }
    
    resp = requests.post(url, headers=headers, json=data, timeout=120)
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content']


def main():
    # 测试：对一家公司做完整尽调
    company_name = "达梦数据"          # ← 可以改成任意公司名
    our_product = "企业级数据库安全与运维解决方案"   # ← 改成你的产品
    
    # 1. 联网搜索
    intel = research_company(company_name)
    
    if not intel:
        print("❌ 没有搜索到情报，请检查Tavily API Key是否正确、网络是否正常")
        return
    
    # 2. AI生成深度报告
    print(f"\n🤖 DeepSeek正在综合{len(intel)}条情报，生成深度尽调报告...")
    report = generate_deep_report(company_name, intel, our_product)
    
    # 3. 保存报告
    report_file = os.path.join(REPORT_DIR, f"{company_name}_深度尽调报告.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# {company_name} 深度尽调报告\n\n")
        f.write(f"我方产品：{our_product}\n\n")
        f.write(report)
    
    # 同时保存原始情报
    intel_file = os.path.join(REPORT_DIR, f"{company_name}_原始情报.json")
    with open(intel_file, 'w', encoding='utf-8') as f:
        json.dump(intel, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ 尽调完成！")
    print(f"  深度报告：{report_file}")
    print(f"  原始情报：{intel_file}")
    print("=" * 60)
    print("\n报告预览：\n")
    print(report[:1500])


if __name__ == '__main__':
    main()
