"""
客户需求信号识别器
核心理念：ToB销售以需求为核心，从公开财务/业务数据中挖掘需求信号
需求类型：扩张型/降本型/转型型/合规型/技术升级型/建设型
运行：python src/demand_analyzer.py
"""
import pandas as pd
import numpy as np
import os
import re

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'company_full_data.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'outputs', 'demand_analysis_result.csv')


def parse_percent(val):
    """百分比字符串转数值"""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).replace('%', '').replace(',', '').strip()
    try:
        return float(val)
    except:
        return None


def identify_demand_signals(row):
    """
    识别单家公司的需求信号
    返回：需求类型列表、需求强度、需求信号描述列表、销售切入建议
    """
    signals = []       # 需求信号
    demand_types = []  # 需求类型
    score = 0          # 需求强度分（0-100）
    
    # 解析财务数据
    revenue_growth = parse_percent(row.get('营收同比增长率'))
    profit_growth = parse_percent(row.get('净利润同比增长率'))
    gross_margin = parse_percent(row.get('毛利率'))
    net_margin = parse_percent(row.get('净利率'))
    debt_ratio = parse_percent(row.get('资产负债率'))
    
    main_business = str(row.get('主营业务', '') or '')
    highlights = str(row.get('公司亮点', '') or '')
    industry = str(row.get('所属行业', '') or '')
    all_text = main_business + highlights + industry
    
    # ========== 1. 扩张型需求 ==========
    if revenue_growth is not None:
        if revenue_growth > 30:
            signals.append(f'营收高速增长{revenue_growth:.0f}%，业务扩张期，系统扩容需求强')
            demand_types.append('扩张型')
            score += 20
        elif revenue_growth > 15:
            signals.append(f'营收稳健增长{revenue_growth:.0f}%，有扩容升级需求')
            demand_types.append('扩张型')
            score += 12
    
    if profit_growth is not None and profit_growth > 30:
        signals.append(f'净利润高增长{profit_growth:.0f}%，经营向好，预算宽松')
        score += 10
    
    # ========== 2. 降本增效型需求 ==========
    if gross_margin is not None and gross_margin < 30:
        signals.append(f'毛利率仅{gross_margin:.0f}%，成本压力大，降本增效需求强')
        demand_types.append('降本型')
        score += 15
    elif gross_margin is not None and gross_margin < 45:
        signals.append(f'毛利率{gross_margin:.0f}%，有一定成本优化空间')
        score += 8
    
    if net_margin is not None and net_margin < 5:
        signals.append(f'净利率仅{net_margin:.0f}%，盈利能力弱，对省钱方案敏感')
        demand_types.append('降本型')
        score += 12
    
    if debt_ratio is not None and debt_ratio > 60:
        signals.append(f'资产负债率{debt_ratio:.0f}%，财务压力大，倾向订阅制/轻资产方案')
        demand_types.append('降本型')
        score += 8
    
    # ========== 3. 数字化/业务转型型需求 ==========
    transform_keywords = {
        '人工智能': 'AI人工智能', 'AI': 'AI人工智能', '大模型': '大模型',
        '云计算': '云计算', '云服务': '云计算', '数字化': '数字化转型',
        '大数据': '大数据', '物联网': '物联网', '工业互联网': '工业互联网',
        '智能制造': '智能制造', '信创': '信创国产化', '国产替代': '国产替代',
        '自主可控': '自主可控', '数字孪生': '数字孪生'
    }
    matched_transform = []
    for kw, label in transform_keywords.items():
        if kw in all_text and label not in matched_transform:
            matched_transform.append(label)
    
    if matched_transform:
        signals.append(f'业务涉及{ "、".join(matched_transform[:3]) }，存在技术升级/转型需求')
        demand_types.append('转型型')
        score += min(20, len(matched_transform) * 8)
    
    # ========== 4. 合规/政策驱动型需求 ==========
    compliance_industries = {
        '金融': '金融行业强监管，数据安全/合规是刚需',
        '银行': '银行业强监管，等保合规刚需',
        '证券': '证券行业强监管，合规需求强',
        '保险': '保险行业合规需求',
        '医疗': '医疗行业数据合规需求',
        '医药': '医药行业合规需求',
        '军工': '军工行业自主可控/保密需求强',
        '国防': '国防军工自主可控需求',
        '政务': '政务行业信创/安全刚需',
        '电力': '电力行业关键信息基础设施保护需求',
        '能源': '能源行业工控安全需求',
        '交通': '交通行业关键基础设施需求'
    }
    for kw, desc in compliance_industries.items():
        if kw in all_text:
            signals.append(desc)
            demand_types.append('合规型')
            score += 15
            break
    
    # 数据安全/网络安全相关业务
    security_keywords = ['网络安全', '数据安全', '信息安全', '密码', '等保', '加密']
    for kw in security_keywords:
        if kw in all_text:
            signals.append('自身涉及安全业务，对安全产品有专业判断力和采购需求')
            demand_types.append('合规型')
            score += 10
            break
    
    # ========== 5. 技术升级型需求 ==========
    tech_keywords = ['研发', '技术', '专利', '软件', '芯片', '算法', '实验室']
    tech_count = sum(1 for kw in tech_keywords if kw in all_text)
    if tech_count >= 3:
        signals.append('技术驱动型公司（研发/专利/算法关键词密集），愿意为新技术付费')
        demand_types.append('技术升级型')
        score += 10
    
    # ========== 6. 建设型需求（新公司） ==========
    found_year = row.get('成立年限')
    if pd.notna(found_year):
        try:
            years = float(found_year)
            if years < 5:
                signals.append(f'成立仅{years:.0f}年，处于IT系统建设期，采购需求旺盛')
                demand_types.append('建设型')
                score += 12
            elif years < 10:
                signals.append(f'成立{years:.0f}年，系统进入升级换代周期')
                score += 6
        except:
            pass
    
    # ========== 需求强度分级 ==========
    score = min(100, score)
    if score >= 50:
        demand_level = '强需求'
    elif score >= 30:
        demand_level = '中需求'
    elif score >= 15:
        demand_level = '弱需求'
    else:
        demand_level = '需求不明显'
    
    # 去重需求类型
    demand_types = list(dict.fromkeys(demand_types))
    
    # 销售切入建议
    suggestions = []
    if '扩张型' in demand_types:
        suggestions.append('以"支撑业务快速扩张、避免系统拖后腿"为切入点')
    if '降本型' in demand_types:
        suggestions.append('以"降本增效、ROI量化、多久回本"为切入点')
    if '转型型' in demand_types:
        suggestions.append('以"助力数字化/AI转型、行业标杆案例"为切入点')
    if '合规型' in demand_types:
        suggestions.append('以"满足监管要求、规避合规风险"为切入点')
    if '技术升级型' in demand_types:
        suggestions.append('以"技术先进性、架构演进、与现有技术栈集成"为切入点')
    if '建设型' in demand_types:
        suggestions.append('以"一站式建设、快速上线、随业务成长扩展"为切入点')
    if not suggestions:
        suggestions.append('需求信号不明显，建议先做需求调研，寻找业务痛点')
    
    return pd.Series({
        '需求类型': '、'.join(demand_types) if demand_types else '待挖掘',
        '需求强度分': score,
        '需求强度等级': demand_level,
        '需求信号数量': len(signals),
        '需求信号详情': ' || '.join(signals) if signals else '暂无明显需求信号',
        '销售切入建议': ' || '.join(suggestions)
    })


def main():
    print("=" * 60)
    print("客户需求信号识别器（需求导向分析）")
    print("=" * 60)
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 未找到输入文件：{INPUT_FILE}")
        return
    
    df = pd.read_csv(INPUT_FILE)
    print(f"📋 共 {len(df)} 家公司待分析")
    
    # 识别需求信号
    print("\n🚀 开始识别需求信号...")
    demand_result = df.apply(identify_demand_signals, axis=1)
    result = pd.concat([df, demand_result], axis=1)
    
    # 保存
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    # 统计
    print("\n" + "=" * 60)
    print("✅ 需求分析完成！")
    print(f"  结果保存：{OUTPUT_FILE}")
    print("=" * 60)
    
    print("\n📊 需求强度分布：")
    print(result['需求强度等级'].value_counts())
    
    print("\n📊 需求类型分布（一家公司可能有多种需求）：")
    all_types = []
    for t in result['需求类型']:
        all_types.extend(t.split('、'))
    print(pd.Series(all_types).value_counts())
    
    print("\n🔥 需求强度TOP10客户：")
    top10 = result.sort_values('需求强度分', ascending=False).head(10)
    for _, row in top10.iterrows():
        print(f"  {row['需求强度分']}分 | {row['需求强度等级']} | {row['公司名称']} | {row['需求类型']}")


if __name__ == '__main__':
    main()
