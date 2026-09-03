"""
客户评分模型模块
功能：数据清洗 → 特征工程（12维度）→ 加权评分 → A/B/C/D分级 → 模型验证
"""
import pandas as pd
import numpy as np
import os
import re
import sys

# 导入 AHP 权重模块（与本文件同目录，兼容从项目根目录运行）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ahp_weights import get_ahp_weights, get_cr_report

# 数据路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_percent(value):
    """把'60.33%'转成60.33"""
    if pd.isna(value) or value == '' or value is None:
        return np.nan
    value = str(value).replace('%', '').replace(',', '').strip()
    try:
        return float(value)
    except:
        return np.nan


def parse_money(value):
    """把'5000万元'转成5000"""
    if pd.isna(value) or value == '' or value is None:
        return np.nan
    value = str(value).replace('元', '').replace(',', '').strip()
    try:
        if '万' in value:
            return float(value.replace('万', ''))
        elif '亿' in value:
            return float(value.replace('亿', '')) * 10000
        else:
            return float(value) / 10000  # 元转万元
    except:
        return np.nan


def calculate_company_age(establish_date):
    """计算公司成立年限"""
    if pd.isna(establish_date) or establish_date == '':
        return np.nan
    try:
        year = int(str(establish_date)[:4])
        return 2026 - year
    except:
        return np.nan


def is_shenzhen(address):
    """判断是否在深圳/华南地区"""
    if pd.isna(address) or address == '':
        return 0
    address = str(address)
    south_china = ['深圳', '广州', '东莞', '佛山', '珠海', '惠州', '中山', '广东', '华南']
    for city in south_china:
        if city in address:
            return 1
    return 0


def is_high_growth_industry(industry, business):
    """判断是否高景气行业（AI/云计算/半导体/网络安全等）"""
    text = str(industry) + str(business)
    hot_keywords = ['人工智能', 'AI', '云计算', '大数据', '半导体', '芯片', '网络安全',
                     '信息安全', '物联网', '5G', '通信', '软件', '数据中心', '工业互联网',
                     '机器人', '自动驾驶', '新能源', '储能', '光伏']
    for kw in hot_keywords:
        if kw in text:
            return 1
    return 0


def calculate_decision_chain_complexity(scale_score):
    """基于规模推断决策链复杂度（规模越大越复杂，得分越低）"""
    # 决策链简单的公司（小公司）得分高，因为决策快
    if pd.isna(scale_score):
        return 50
    return max(0, min(100, 100 - scale_score * 0.5))


def calculate_demand_match(business, company_highlight):
    """基于主营业务和亮点推断需求匹配度（科技公司对IT解决方案需求高）"""
    text = str(business) + str(company_highlight)
    score = 50  # 基础分
    high_demand_keywords = ['企业级', '解决方案', '数字化', '信息化', '云', '数据', '智能',
                            '网络', '安全', '系统', '软件', '服务', '咨询', '管理']
    for kw in high_demand_keywords:
        if kw in text:
            score += 5
    return min(100, score)


def main():
    print("=" * 70)
    print("📊 客户评分模型")
    print("=" * 70)
    
    # 1. 读取数据
    input_file = os.path.join(DATA_DIR, 'company_full_data.csv')
    if not os.path.exists(input_file):
        print(f"❌ 找不到文件：{input_file}")
        return
    
    df = pd.read_csv(input_file)
    print(f"✅ 读取到 {len(df)} 家公司数据")
    
    # 2. 数据清洗
    print("\n🔧 正在进行数据清洗...")
    
    # 转换百分比字段
    for col in ['毛利率', '净利率', '营收同比增长率', '净利润同比增长率', '资产负债率']:
        df[col + '_数值'] = df[col].apply(parse_percent)
    
    # 转换注册资本
    df['注册资本_万元'] = df['注册资本'].apply(parse_money)
    
    # 计算成立年限
    df['成立年限'] = df['成立日期'].apply(calculate_company_age)
    
    # 地域匹配
    df['华南地区'] = df['办公地址'].apply(is_shenzhen)
    
    # 高景气行业
    df['高景气行业'] = df.apply(lambda x: is_high_growth_industry(x['所属行业'], x['主营业务']), axis=1)
    
    print("✅ 数据清洗完成")
    
    # 3. 特征工程：12个维度评分（0-100分）
    print("\n🎯 正在进行特征工程（12维度评分）...")
    
    # 维度1：企业规模（基于注册资本，对数标准化）
    df['规模得分'] = df['注册资本_万元'].apply(
        lambda x: min(100, np.log10(x + 1) * 20) if pd.notna(x) else 50
    )
    
    # 维度2：增长能力（营收增长率+净利润增长率）
    df['增长得分'] = df.apply(
        lambda x: np.nanmean([
            min(100, max(0, x['营收同比增长率_数值'] * 2 + 50)),
            min(100, max(0, x['净利润同比增长率_数值'] * 2 + 50))
        ]) if pd.notna(x['营收同比增长率_数值']) or pd.notna(x['净利润同比增长率_数值']) else 50,
        axis=1
    )
    
    # 维度3：盈利能力（毛利率+净利率）
    df['盈利得分'] = df.apply(
        lambda x: np.nanmean([
            min(100, max(0, x['毛利率_数值'] * 1.5)),
            min(100, max(0, x['净利率_数值'] * 2 + 50))
        ]) if pd.notna(x['毛利率_数值']) or pd.notna(x['净利率_数值']) else 50,
        axis=1
    )
    
    # 维度4：财务健康度（资产负债率越低越健康）
    df['财务健康得分'] = df['资产负债率_数值'].apply(
        lambda x: min(100, max(0, 100 - x * 1.2)) if pd.notna(x) else 50
    )
    
    # 维度5：技术投入（基于公司亮点和主营业务中的技术关键词）
    df['技术投入得分'] = df.apply(
        lambda x: calculate_demand_match(x['主营业务'], x['公司亮点']),
        axis=1
    )
    
    # 维度6：企业成熟度（成立年限，10-20年最佳）
    df['成熟度得分'] = df['成立年限'].apply(
        lambda x: min(100, max(0, 100 - abs(x - 15) * 5)) if pd.notna(x) else 50
    )
    
    # 维度7：上市年限（上市越久治理越规范）
    df['上市年限得分'] = df['上市日期'].apply(
        lambda x: min(100, max(0, (2026 - int(str(x)[:4])) * 5)) if pd.notna(x) and str(x)[:4].isdigit() else 50
    )
    
    # 维度8：地域匹配（华南地区加分）
    df['地域得分'] = df['华南地区'].apply(lambda x: 100 if x == 1 else 50)
    
    # 维度9：行业景气度（高景气行业加分）
    df['行业景气得分'] = df['高景气行业'].apply(lambda x: 100 if x == 1 else 50)
    
    # 维度10：企业性质（国企预算稳定加分，这里简化处理）
    df['企业性质得分'] = 60  # 默认中性分，后续可优化
    
    # 维度11：决策链复杂度（小公司决策快加分）
    df['决策链得分'] = df['规模得分'].apply(calculate_decision_chain_complexity)
    
    # 维度12：需求匹配度
    df['需求匹配得分'] = df.apply(
        lambda x: calculate_demand_match(x['主营业务'], x['公司亮点']),
        axis=1
    )
    
    print("✅ 12维度特征工程完成")
    
    # 4. 加权评分（AHP层次分析法确定权重）
    print("\n⚖️ 正在进行加权评分...")
    
        # 权重由 AHP 层次分析法确定（判断矩阵→特征向量→一致性检验→层次总排序）
    weights = get_ahp_weights()
    print("AHP 一致性检验：")
    print(get_cr_report())
    print("AHP 12维权重：", {k: round(v, 4) for k, v in weights.items()})

    # 计算总分
    df['总分'] = 0
    for col, weight in weights.items():
        df['总分'] += df[col] * weight
    
    df['总分'] = df['总分'].round(2)
    
    # 5. A/B/C/D分级
    print("\n📋 正在进行客户分级...")
    
    def grade_customer(score):
        if score >= 75:
            return 'A级（高价值优先跟进）'
        elif score >= 65:
            return 'B级（重点跟进）'
        elif score >= 55:
            return 'C级（常规跟进）'
        else:
            return 'D级（低优先级/观察）'
    
    df['客户分级'] = df['总分'].apply(grade_customer)
    
    # 统计分级分布
    grade_counts = df['客户分级'].value_counts()
    print("分级分布：")
    for grade in ['A级（高价值优先跟进）', 'B级（重点跟进）', 'C级（常规跟进）', 'D级（低优先级/观察）']:
        count = grade_counts.get(grade, 0)
        print(f"  {grade}：{count} 家（{count/len(df)*100:.1f}%）")
    
    # 6. 保存结果
    output_file = os.path.join(OUTPUT_DIR, 'customer_scoring_result.csv')
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # 保存TOP50高价值客户
    top50 = df.sort_values('总分', ascending=False).head(50)
    top50_file = os.path.join(OUTPUT_DIR, 'top50_high_value_customers.csv')
    top50.to_csv(top50_file, index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 70)
    print(f"✅ 评分模型运行完成！")
    print(f"   - 评分公司总数：{len(df)} 家")
    print(f"   - 平均分：{df['总分'].mean():.2f}")
    print(f"   - 最高分：{df['总分'].max():.2f}")
    print(f"   - 最低分：{df['总分'].min():.2f}")
    print(f"   - A级客户：{grade_counts.get('A级（高价值优先跟进）', 0)} 家")
    print(f"   - 结果保存路径：{OUTPUT_DIR}")
    print("=" * 70)
    print("\n📁 生成的文件：")
    print(f"   1. customer_scoring_result.csv - 全部客户评分结果")
    print(f"   2. top50_high_value_customers.csv - TOP50高价值客户")
    
    # 显示TOP10
    print("\n🏆 TOP10高价值客户：")
    top10 = df.sort_values('总分', ascending=False).head(10)
    for idx, row in top10.iterrows():
        print(f"  {row['总分']:.1f}分 | {row['客户分级'][:2]} | {row['公司名称']}({row['股票代码']}) | {row['所属行业'][:20] if pd.notna(row['所属行业']) else '未知'}")


if __name__ == '__main__':
    main()
